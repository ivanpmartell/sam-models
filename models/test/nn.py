import sys
import os
import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from nn_models import select_model, select_device

def parse_commandline():
    parser = argparse.ArgumentParser(description='Neural network testing script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where scoring metric results will be saved. Leave empty to use params directory')
    parser.add_argument('--model', type=str, required=True,
                    help='Type of neural network model to use')
    parser.add_argument('--params', type=str, required=True,
                    help='Pretrained parameters file')
    return parser.parse_args()

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)*1024
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(2) == y.argmax(2)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    test_acc = 100*correct
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_acc

def commands(args, X, y):
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    ds = TensorDataset(tensor_x,tensor_y)
    train_dataloader = DataLoader(ds, batch_size=1)
    model = args.NNModel().to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(args.params))
    return test(train_dataloader, model, loss_fn, args.device)

def main():
    args = parse_commandline()
    args.NNModel = select_model(args.model)
    args.device = select_device()
    work_on_testing(args, commands)

main()