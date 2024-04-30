import sys
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Optional app description')
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
                        help='Keyword or Comma separated list of methods to include in majority consensus prediction. Keywords: all, top, avg, low',
                        default="all")
    return parser.parse_args()

def preprocess(args, predictions, assignment):
    classes = list(get_ss_q8())
    freqs = np.zeros((1024,len(classes)))
    for i in range(len(assignment)):
        for prediction in predictions.values():
            freqs[i, ss_index(prediction[i])] += 1
    max_class_freqs = freqs.max(axis=0)
    normalized_freqs = np.divide(freqs, max_class_freqs, out=np.zeros_like(freqs), where=max_class_freqs!=0)
    return normalized_freqs, onehot_encode(assignment.seq)

def data_process(args, df, out_paths):
    X_train, X_test, y_train, y_test = train_test_split(df.x, df.y, test_size=0.20)
    write_npz(out_paths["train"], X_train, y_train)
    write_npz(out_paths["test"], X_test, y_test)
    write_npz(out_paths["all"], df.x, df.y)
    
def main():
    args = parse_commandline()
    predictors = choose_methods(args.methods)
    work_on_all_data(args, predictors, preprocess, data_process)

main()