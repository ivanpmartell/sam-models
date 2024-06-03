import sys
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Script to preprocess data for machine learning use')
    parser.add_argument('dir', type=str,
                    help='Input directory containing clusters')
    parser.add_argument('--out_dir', type=str,
                    help='Output directory to save numpy data. Ignore to use input directory')
    parser.add_argument('--pred_ext', type=str,
                        help='Extension of ss prediction files',
                        default=".sspfa")
    parser.add_argument('--assign_ext', type=str,
                        help='Extension of ss assigned files',
                        default=".ssfa")
    parser.add_argument('--methods', type=str,
                        help='Keyword or Comma separated list of methods to include in prediction. Keywords: all, top, avg, low',
                        default="all")
    parser.add_argument('--split_type', type=str,
                        help='Method of splitting the data. Keywords: kfold, train_test',
                        default="train_test")
    parser.add_argument('--split_size', type=str,
                        help='Amount of data to split. Values: kfold=1 to 100, train_test=0 to 1',
                        default="0.20")
    return parser.parse_args()

def train_test(df, out_paths, split):
    X_train, X_test, y_train, y_test = train_test_split(df.x, df.y, test_size=split)
    write_npz(out_paths["train"], X_train, y_train)
    write_npz(out_paths["test"], X_test, y_test)
    write_npz(out_paths["all"], df.x, df.y)

def kfold(df, out_paths, split):
    data_len = len(df.x)
    usable_split = closest_divisor(data_len, split)
    if usable_split != split:
        exit(2)
    Xs  = np.split(df.x, usable_split)
    Ys = np.split(df.y, usable_split)
    for i in range(usable_split):
        write_npz(out_paths["kfold"].format(i), Xs[i], Ys[i])

def data_process(args, df, out_paths):
    shuffled = df.sample(frac=1).reset_index(drop=True)
    if args.split_type == "kfold":
        kfold(shuffled, out_paths, int(args.split_size))
    elif args.split_type == "train_test":
        train_test(shuffled, out_paths, float(args.split_size))
    else:
        print("Uknown splitting method")
        exit(1)

def main():
    args = parse_commandline()
    predictors = choose_methods(args.methods)
    work_on_data(args, predictors, data_process)

main()