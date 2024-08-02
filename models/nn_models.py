import math
import torch
import lightning as L
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

class FCNN(nn.Module):
    def __init__(self, num_predictors, classes=9, seq_len=1024):
        super(FCNN, self).__init__()
        self.classes = classes
        self.seq_len = seq_len
        self.hidden = nn.Sequential(nn.Linear(seq_len*num_predictors*classes, seq_len*classes),
                                    nn.ReLU(),
                                    nn.Linear(seq_len*classes, seq_len),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.out = nn.Linear(seq_len, seq_len*classes)

    def forward(self, x):
        bsize = x.shape[0]
        x = x.reshape(bsize, -1)
        x = self.hidden(x)
        x = self.out(x).reshape(bsize, self.classes, self.seq_len)
        return x
    
class CNN(nn.Module):
    def __init__(self, num_predictors, classes=9, seq_len=1024):
        super(CNN, self).__init__()
        self.classes = classes
        self.predictors = num_predictors
        self.seq_len = seq_len
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=classes,
                                            out_channels=classes, kernel_size=(5,17), padding='same'),
                                    nn.ReLU(),
                                    nn.Dropout(0.2))
        self.hidden = nn.Sequential(nn.Linear(classes*seq_len*num_predictors, seq_len*classes),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.out = nn.Linear(seq_len*classes, seq_len*classes)

    def forward(self, x):
        bsize = x.shape[0]
        features = self.conv1(x)
        features = features.reshape(features.shape[0], -1)
        hidden = self.hidden(features)
        out = self.out(hidden).reshape(bsize, self.classes, self.seq_len)
        return out
    
class RNN(nn.Module):
    def __init__(self, num_predictors, classes=9, seq_len=1024):
        super(RNN, self).__init__()
        self.classes = classes
        self.seq_len = seq_len
        self.predictors = num_predictors
        self.bilstm = nn.LSTM(
            seq_len, 32, 2, bias=True,
            batch_first=True, dropout=0.5, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(32*2*classes*num_predictors, seq_len*classes),
                                 nn.ReLU(),
                                 nn.Dropout(0.5))
        self.fc2 = nn.Linear(seq_len*classes, seq_len*classes)

    def forward(self, x):
        bsize = x.shape[0]
        rnns =[]
        for i in range(self.predictors):
            rnn, _ = self.bilstm(x[:,:,:,i])
            rnns.append(rnn)
        x = torch.stack(rnns, dim=-1)
        x = self.fc1(torch.flatten(x, 1))
        x = self.fc2(x).squeeze(dim=-1).reshape(bsize, self.classes, self.seq_len)
        return x
    
class TNN(nn.Module):
    def __init__(self, num_predictors, classes=9, seq_len=1024, d_model = 32, nhead = 2, d_hid = 128,
                 nlayers = 2, dropout = 0.5):
        super(TNN, self).__init__()
        self.pos_encoder = PosEncoding(d_model, dropout, seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(classes, d_model)
        self.linear = nn.Linear(d_model, classes)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, src_mask = None):
        src = x.argmax(1)
        src_list = []
        for i in range(x.shape[-1]):
            src_list.append(src[:,:,i])
        out_list = []
        for src in src_list:
            src = self.embedding(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            if src_mask is None:
                src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1], device=src.get_device())
            output = self.transformer_encoder(src, src_mask)
            out_list.append(self.linear(output))
        out = torch.zeros_like(out_list[0])
        for i in range(x.shape[-1]):
            out += out_list[i]
        return out.permute(0,2,1)

class PosEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super(PosEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(0)
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        out = x + self.pe[:x.size(1)].to(x.get_device())
        return self.dropout(out)

def select_model(model: str) -> callable:
    model = model.lower()
    if "fullyconnected" in model:
        return FCNN
    elif "convolutional" in model:
        return CNN
    elif "recurrent" in model:
        return RNN
    elif "transformer" in model:
        return TNN
    else:
        print("Incorrect model, please choose from: FullyConnected, Convolutional, Recurrent, Transformer")
        exit(1)

def select_device(print_device = True):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if print_device:
        print(f"Using {device} device")
    return device

class CustomDataset(Dataset):
    def __init__(self, x, y, x_transform=None, y_transform=None):
        self.x = x
        self.y = y
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, index):
        x = self.x[index]
        if self.x_transform:
            x = self.x_transform(x)
        y = self.y[index]
        if self.y_transform:
            y = self.y_transform(y)
            y = torch.Tensor(y)
        else:
            y = torch.LongTensor(y)
        return torch.Tensor(x), y

    def __len__(self):
        return len(self.y)

class LitModel(L.LightningModule):
    def __init__(self, nnModel, win_size, max_len, predictors):
        super().__init__()
        self.nnModel = nnModel(predictors, seq_len=win_size)
        self.win_size = win_size
        self.max_len = max_len

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nnModel.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1)
        y_hat = []
        for i in range(self.max_len // self.win_size):
            input_cut = x[:,:,i*self.win_size:(i+1)*self.win_size]
            completely_masked = torch.zeros_like(input_cut)
            completely_masked[:,0,:,:] = 1
            if torch.equal(input_cut, completely_masked):
                empty_chunk = torch.zeros_like(y_hat[0])
                empty_chunk[:,0,:] = 1
                y_hat.append(empty_chunk)
            else:
                y_hat.append(self.nnModel(input_cut))
        y_hat = torch.cat(y_hat, 2)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        if y.dtype is torch.long:
            target = y
        else:
            target = y.argmax(1)
        acc = (y_hat.argmax(1) == target).type(torch.float).mean().item()
        self.log("train_acc", acc, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1)
        y_hat = []
        seq_chunk = self.win_size
        for i in range(self.max_len // seq_chunk):
            input_cut = x[:,:,i*seq_chunk:(i+1)*seq_chunk]
            completely_masked = torch.zeros_like(input_cut)
            completely_masked[:,0,:,:] = 1
            if torch.equal(input_cut, completely_masked):
                empty_chunk = torch.zeros_like(y_hat[0])
                empty_chunk[:,0,:] = 1
                y_hat.append(empty_chunk)
            else:
                y_hat.append(self.nnModel(input_cut))
        y_hat = torch.cat(y_hat, 2)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True)
        if y.dtype is torch.long:
            target = y
        else:
            target = y.argmax(1)
        acc = (y_hat.argmax(1) == target).type(torch.float).mean().item()
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(-1)
        y_hat = []
        seq_chunk = self.win_size
        for i in range(self.max_len // seq_chunk):
            input_cut = x[:,:,i*seq_chunk:(i+1)*seq_chunk]
            completely_masked = torch.zeros_like(input_cut)
            completely_masked[:,0,:,:] = 1
            if torch.equal(input_cut, completely_masked):
                empty_chunk = torch.zeros_like(y_hat[0])
                empty_chunk[:,0,:] = 1
                y_hat.append(empty_chunk)
            else:
                y_hat.append(self.nnModel(input_cut))
        y_hat = torch.cat(y_hat, 2)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss, on_epoch=True)
        if y.dtype is torch.long:
            target = y
        else:
            target = y.argmax(1)
        acc = (y_hat.argmax(1) == target).type(torch.float).mean().item()
        self.log("test_acc", acc, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = []
        seq_chunk = self.win_size
        for i in range(self.max_len // seq_chunk):
            input_cut = x[:,:,i*seq_chunk:(i+1)*seq_chunk]
            completely_masked = torch.zeros_like(input_cut)
            completely_masked[:,0,:,:] = 1
            if torch.equal(input_cut, completely_masked):
                empty_chunk = torch.zeros_like(y_hat[0])
                empty_chunk[:,0,:] = 1
                y_hat.append(empty_chunk)
            else:
                y_hat.append(self.nnModel(input_cut))
        return torch.cat(y_hat, 2)
    
    def forward(self, x):
        return self.nnModel(x)