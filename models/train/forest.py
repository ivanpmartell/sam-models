import sys
import os
import argparse
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
import numpy as np

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Random forest training script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    parser.add_argument('--model', type=str, default="forest",
                    help='Name to use for this model')
    parser.add_argument('--preprocess', type=str, default="frequency_max_location",
                    help='Type of preprocessing for input data. Choices: nominal_location, frequency_location, frequency_max_location')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length of inputs.')
    return parser.parse_args()

#For X: use any location preprocesses or nominal_windowed
def transform_data(X, y, preprocess, max_len):
    X = preprocess(X, max_len)
    y = single_target_preprocess(y)
    return X, y

def commands(args, X, y):
    X, y = transform_data(X, y, args.preprocess, args.seq_len)
    extra_tree = ExtraTreeClassifier()
    cls = BaggingClassifier(extra_tree).fit(X, y)
    write_classifier(args.out_file, cls)

def main():
    args = parse_commandline()
    args.preprocess = choose_preprocess(args.preprocess)
    work_on_training(args, commands)

main()