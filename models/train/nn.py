import sys
import os
import argparse
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from data_preprocess import onehot_preprocess
from nn_models import select_model, CustomDataset, LitModel

def parse_commandline():
    parser = argparse.ArgumentParser(description='Neural network training script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    parser.add_argument('--model', type=str, required=True,
                    help='Type of neural network model to use')
    parser.add_argument('--predictors', type=int, required=True,
                    help='Amount of predictors in input data')
    parser.add_argument('--win_len', type=int, default=1024,
                    help='Length of the window to be used')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length')
    return parser.parse_args()

def commands(args, X, y):
    out_dir = os.path.dirname(args.out_file)
    out_fname = remove_ckpt_ext(os.path.basename(args.out_file))
    ds = CustomDataset(X, y, x_transform=onehot_preprocess)
    train_set, val_set = random_split(ds, [0.9, 0.1])
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=12)
    val_dataloader = DataLoader(val_set)
    l_model = LitModel(args.NNModel, args.win_len, args.seq_len, args.predictors)
    trainer = L.Trainer(default_root_dir=f"{out_dir}/nn/",
                        max_epochs=100,
                        callbacks=[EarlyStopping(monitor="val_acc", mode="max", patience=10,  min_delta=0.0001),
                                   ModelCheckpoint(monitor='val_acc',  mode="max", save_top_k=1, dirpath=out_dir, filename=out_fname)],
                        num_sanity_val_steps=2)
    trainer.fit(model=l_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def main():
    args = parse_commandline()
    args.NNModel = select_model(args.model)
    work_on_training(args, commands)

main()