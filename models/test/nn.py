import sys
import os
import argparse
import lightning as L
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from data_preprocess import onehot_preprocess
from nn_models import select_model, CustomDataset, LitModel

def parse_commandline():
    parser = argparse.ArgumentParser(description='Neural network testing script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where scoring metric results will be saved. Leave empty to use params directory')
    parser.add_argument('--model', type=str, required=True,
                    help='Type of neural network model to use')
    parser.add_argument('--params', type=str, required=True,
                    help='Pretrained parameters (checkpoint) file')
    parser.add_argument('--predictors', type=int, required=True,
                    help='Amount of predictors in input data')
    parser.add_argument('--win_len', type=int, default=1024,
                    help='Length of the window to be used')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length')
    return parser.parse_args()

def commands(args, X, y):
    out_dir = os.path.dirname(args.out_file)
    ds = CustomDataset(X, y, x_transform=onehot_preprocess)
    test_dataloader = DataLoader(ds, batch_size=1)
    model = args.NNModel(args.predictors, seq_len=args.win_len)
    trained_model = LitModel.load_from_checkpoint(args.params, nnModel=model, win_size=args.win_len, max_len=args.seq_len)
    trainer = L.Trainer(default_root_dir=f"{out_dir}/nn/", num_sanity_val_steps=0)
    return trainer.test(model=trained_model, dataloaders=test_dataloader)

def main():
    args = parse_commandline()
    args.NNModel = select_model(args.model)
    work_on_testing(args, commands)

main()