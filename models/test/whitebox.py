import sys
import os
import argparse
import pickle
from interpret import show, set_visualize_provider
from interpret.provider import DashProvider
set_visualize_provider(DashProvider.from_address(('127.0.0.1', 7001)))

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from data_preprocess import single_target_preprocess, choose_preprocess

def parse_commandline():
    parser = argparse.ArgumentParser(description='Whitebox testing script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where scoring metric results will be saved. Leave empty to use params directory')
    parser.add_argument('--model', type=str, default="ebm",
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
    test_acc = cls.score(X, np.char.mod('%d.0', y))
    print(f"Accuracy: {test_acc}")
    show(cls.explain_global())
    return test_acc

def main():
    args = parse_commandline()
    args.model = args.model.lower()
    args.preprocess = choose_preprocess(args.preprocess)
    work_on_testing(args, commands)

main()
input("Press Enter to exit...")