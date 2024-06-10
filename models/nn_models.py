import math
import torch
from torch import nn
import torch.nn.functional as F
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
                                    nn.Dropout(0.1))
        self.out = nn.Linear(seq_len, seq_len*classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        hidden = self.hidden(x)
        out = self.out(hidden).reshape(x.shape[0], self.seq_len, self.classes)
        return out
    
class CNN(nn.Module):
    def __init__(self, num_predictors, classes=9, seq_len=1024):
        super(CNN, self).__init__()
        self.classes = classes
        self.seq_len = seq_len
        self.conv = nn.Sequential(nn.Conv1d(in_channels=classes,
                                            out_channels=classes, kernel_size=21),
                                    nn.ReLU(),
                                    nn.MaxPool1d(4),
                                    nn.Dropout(0.5))
        self.hidden = nn.Sequential(nn.Linear(classes*251, seq_len),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.out = nn.Linear(seq_len, seq_len*classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.conv(x).squeeze(dim=-1)
        features = features.reshape(features.shape[0], -1)
        hidden = self.hidden(features)
        out = self.out(hidden).reshape(x.shape[0], self.seq_len, self.classes)
        return out
    
class RNN(nn.Module):
    def __init__(self, num_predictors, classes=9, seq_len=1024):
        super(RNN, self).__init__()
        self.classes = classes
        self.seq_len = seq_len
        self.bilstm = nn.LSTM(
            classes, 32, 2, bias=True,
            batch_first=True, dropout=0.5, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(32*2*seq_len, seq_len),
                                 nn.ReLU(),
                                 nn.Dropout(0.5))
        self.fc2 = nn.Linear(seq_len, seq_len*classes)

    def forward(self, x):
        rnn, _ = self.bilstm(x)
        hidden = self.fc1(torch.ravel(rnn, 1))
        out = self.fc2(hidden).squeeze(dim=-1).reshape(x.shape[0], self.seq_len, self.classes)
        return out
    
class TNN(nn.Module):
    def __init__(self, num_predictors, classes=9, seq_len=1024, d_model = 32, nhead = 2, d_hid = 128,
                 nlayers = 2, dropout = 0.5):
        super(TNN, self).__init__()
        self.device = select_device(False)
        self.pos_encoder = PosEncoding(d_model, dropout, seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(classes, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, classes)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, src_mask = None):
        src = x.argmax(2)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(self.device)
        output = self.transformer_encoder(src, src_mask)
        out = self.linear(output)
        return out

class PosEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super(PosEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(0)
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        out = x + self.pe[:x.size(0)]
        return self.dropout(out)

def select_model(model):
    model = model.lower()
    if model == "fullyconnected":
        return FCNN
    elif model == "convolutional":
        return CNN
    elif model == "recurrent":
        return RNN
    elif model == "transformer":
        return TNN
    else:
        print("Incorrect model, please choose: FullyConnected, Convolutional, Recurrent, Transformer")
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
        return torch.Tensor(x), torch.Tensor(y)

    def __len__(self):
        return len(self.y)