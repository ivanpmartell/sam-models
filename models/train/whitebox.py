import sys
import os
import argparse
from interpret.glassbox import ExplainableBoostingClassifier, APLRClassifier

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from data_preprocess import single_target_preprocess, choose_preprocess

def parse_commandline():
    parser = argparse.ArgumentParser(description='Random forest training script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    parser.add_argument('--model', type=str, default="ebm",
                    help='Type of tree model to use. Choices: APLR, EBM')
    parser.add_argument('--preprocess', type=str, default="frequency_max_location",
                    help='Type of preprocessing for input data. Choices: nominal_location, frequency_location, frequency_max_location')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length of inputs.')
    parser.add_argument('--win_side_len', type=int,
                    help='Size of window for input.')
    return parser.parse_args()

#For X: use any location preprocesses or nominal_windowed
def transform_data(X, y, preprocess, max_len, win_side_len):
    if win_side_len is not None:
        X = preprocess(X, max_len, win_side_len)
    else:
        X = preprocess(X, max_len)
    y = single_target_preprocess(y)
    return X, y

def commands(args, X, y):
    X, y = transform_data(X, y, args.preprocess, args.seq_len, args.win_side_len)
    if "aplr" in args.model:
        cls = APLRClassifier().fit(X,y)
    elif "ebm" in args.model:
        cls = ExplainableBoostingClassifier().fit(X,y)
    else:
        raise ValueError("Wrong model type. Please choose one of Automatic Piecewise Linear Regression (APLR), Explainable Boosting Machine (EBM). ")
    write_classifier(args.out_file, cls)

def main():
    args = parse_commandline()
    args.model = args.model.lower()
    args.preprocess = choose_preprocess(args.preprocess)
    work_on_training(args, commands)

main()