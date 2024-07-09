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
from nn_models import select_model, select_device, CustomDataset, LitModel, Simplest

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
    return parser.parse_args()

def commands(args, X, y):
    out_dir = os.path.dirname(args.out_file)
    out_fname = os.path.basename(args.out_file)
    ds = CustomDataset(X, y, x_transform=onehot_preprocess, y_transform=onehot_preprocess)
    train_set, val_set = random_split(ds, [0.9, 0.1])
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=4)
    val_dataloader = DataLoader(val_set)
    model = LitModel(args.NNModel(args.predictors))
    trainer = L.Trainer(default_root_dir=f"{out_dir}/nn/",
                        max_epochs=100,
                        callbacks=[EarlyStopping(monitor="val_acc", mode="max", patience=20),
                                   ModelCheckpoint(monitor='val_acc', save_top_k=1, dirpath=out_dir, filename=out_fname)],
                        num_sanity_val_steps=0)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def main():
    args = parse_commandline()
    args.NNModel = select_model(args.model)
    args.device = select_device()
    work_on_training(args, commands)

main()