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
    return parser.parse_args()

#Also use nominal_location_preprocess
def transform_data(X, y, preprocess=frequency_max_location_preprocess):
    X = preprocess(X)
    y = single_target_preprocess(y)
    return X, y

def commands(args, X, y):
    X, y = transform_data(X, y)
    extra_tree = ExtraTreeClassifier()
    cls = BaggingClassifier(extra_tree).fit(X, y)
    write_classifier(args.out_file, cls)

def main():
    args = parse_commandline()
    work_on_training(args, commands)

main()