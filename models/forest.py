import sys
import os
import argparse
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
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
        
    
    return new_X, new_y

def commands(args, X, y):
    X, y = transform_data(X, y)
    extra_tree = ExtraTreeClassifier()
    cls = BaggingClassifier(extra_tree).fit(X, y)
    print(cls.score(X, y))
    prediction = ""
    truth = ""
    q8_ss = get_ss_q8()
    for i in range(485):
        y_hat = cls.predict(X[i].reshape(1, -1))
        prediction += q8_ss[int(y_hat.item())]
        truth += q8_ss[int(y[i].item())]
    print(prediction)
    print(truth)
    print("Done!")

def main():
    args = parse_commandline()
    work_on_training(args, commands)

main()