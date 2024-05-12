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
    parser = argparse.ArgumentParser(description='Random forest training script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    parser.add_argument('--model', type=str, default="forest",
                    help='Name to use for this model')
    return parser.parse_args()

def transform_data(X, y):
    new_X = np.zeros(shape=(len(y)*1024, 1025))
    new_y = np.zeros(shape=(len(y)*1024,))
    for i in range(len(y)):
        max_X = X[i].argmax(1)
        max_y = y[i].argmax(1)
        for j in range(1024):
            new_X[i*1024+j] = np.append(max_X, j)
            new_y[i*1024+j] = max_y[j]
    return new_X, new_y

def commands(args, X, y):
    X, y = transform_data(X, y)
    extra_tree = ExtraTreeClassifier()
    cls = BaggingClassifier(extra_tree).fit(X, y)
    with open(args.out_file, 'wb') as f:
        pickle.dump(cls, f)

def main():
    args = parse_commandline()
    work_on_training(args, commands)

main()