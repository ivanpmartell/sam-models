import sys
import os
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    return parser.parse_args()

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
    
class ConvNet(nn.Module):
    def __init__(self):
      super(ConvNet, self).__init__()
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

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)*485
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(2) == y.argmax(2))[:,:485].type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def commands(args, X, y):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    ds = TensorDataset(tensor_x,tensor_y)
    train_dataloader = DataLoader(ds, batch_size=1)
    model = ConvNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
    test(train_dataloader, model, loss_fn, device)
    print("Done!")

def main():
    args = parse_commandline()
    work_on_training(args, commands)

main()