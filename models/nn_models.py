from torch import nn
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self):
      super(FCNN, self).__init__()
      self.fc1 = nn.Linear(1024*9, 512*9)
      self.fc2 = nn.Linear(512*9, 1024*9)

    def forward(self, x):
      x = x.reshape(x.shape[0], -1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = x.reshape(x.shape[0], 1024, 9)
      output = F.log_softmax(x, dim=2)
      return output
    
class CNN(nn.Module):
    def __init__(self):
      super(CNN, self).__init__()
      self.conv = nn.Sequential(nn.Conv1d(in_channels=9,
                                           out_channels=9, kernel_size=21),
                                 nn.ReLU(),
                                 nn.MaxPool1d(4))
      self.hidden = nn.Sequential(nn.Linear(9*251, 2048),
                                 nn.ReLU())
      self.out = nn.Linear(2048, 1024*9)

    def forward(self, x):
      x = x.permute(0, 2, 1)
      features = self.conv(x).squeeze(dim=-1)
      features = features.reshape(features.shape[0], -1)
      hidden = self.hidden(features)
      out = self.out(hidden).reshape(x.shape[0], 1024, 9)
      out = F.log_softmax(out, dim=2)
      return out
    


def select_model(args):
    args.model = args.model.lower()
    if args.model == "fullyconnected":
        return FCNN
    elif args.model == "convolutional":
        return CNN
    elif args.model == "recurrent":
        return RNN
    elif args.model == "transformer":
        return TNN
    else:
        print("Incorrect model, please choose: FullyConnected, Convolutional, Recurrent, Transformer")
        exit(1)