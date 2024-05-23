import sys
import os
import argparse
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from nn_models import select_model, select_device

def parse_commandline():
    parser = argparse.ArgumentParser(description='Neural network training script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    parser.add_argument('--model', type=str, required=True,
                    help='Type of neural network model to use')
    return parser.parse_args()

def train(dataloader, model, loss_fn, optimizer, best_accuracy, device):
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
        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def evaluate(model, X_test, y_test):
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    return acc

def commands(args, X, y):
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    ds = TensorDataset(tensor_x,tensor_y)
    train_set, test_set = random_split(ds, [0.7, 0.3])
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=16)
    X_test, y_test = default_collate(test_set)
    model = args.NNModel().to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=50)
    epochs = 100
    early_stop_thresh = 5
    best_accuracy = -1
    best_epoch = -1
    for cur_epoch in range(epochs):
        print(f"Epoch {cur_epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, best_accuracy, args.device)
        acc = evaluate(model, X_test, y_test)
        print(f"End of epoch {cur_epoch}: accuracy = {acc:.2f}%")
        if acc > best_accuracy:
            best_accuracy = acc
            best_epoch = cur_epoch
            torch.save(model.state_dict(), args.out_file)
        elif cur_epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {cur_epoch}")
            break
        scheduler.step()

def main():
    args = parse_commandline()
    args.NNModel = select_model(args.model)
    args.device = select_device()
    work_on_training(args, commands)

main()