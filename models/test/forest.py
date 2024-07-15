import sys
import os
import argparse
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
import numpy as np
import pickle

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Random forest testing script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where scoring metric results will be saved. Leave empty to use params directory')
    parser.add_argument('--model', type=str, default="forest",
                    help='Name to use for this model')
    parser.add_argument('--params', type=str, required=True,
                    help='Pretrained parameters (checkpoint) file')
    parser.add_argument('--preprocess', type=str, default="frequency_max_location",
                    help='Type of preprocessing for input data. Choices: nominal_location, frequency_location, frequency_max_location')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length of inputs.')
    parser.add_argument('--win_side_len', type=int,
                    help='Size of window for input.')
    return parser.parse_args()

#Also use nominal_location_preprocess or frequency_max_location_preprocess
def transform_data(X, y, preprocess, max_len, win_side_len):
    if win_side_len is not None:
        X = preprocess(X, max_len, win_side_len)
    else:
        X = preprocess(X, max_len)
    y = single_target_preprocess(y)
    return X, y

def commands(args, X, y):
    X, y = transform_data(X, y, args.preprocess, args.seq_len, args.win_side_len)
    with open(args.params, 'rb') as f:
        cls = pickle.load(f)
    test_acc = cls.score(X, y)
    print(f"Accuracy: {test_acc}")
    return test_acc

def main():
    args = parse_commandline()
    args.preprocess = choose_preprocess(args.preprocess)
    work_on_testing(args, commands)

main()