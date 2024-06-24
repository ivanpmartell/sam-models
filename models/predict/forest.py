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
    parser = argparse.ArgumentParser(description='Mutation secondary structure forest predictor')
    parser.add_argument('dir', type=str,
                    help='Input directory containing clusters')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    parser.add_argument('--methods', type=str,
                        help='Keyword or Comma separated list of methods to include in majority consensus prediction. Keywords: all, top, avg, low',
                        default="all")
    parser.add_argument('--seq_ext', type=str,
                        help='Extension of ss assigned files',
                        default=".fa")
    parser.add_argument('--pred_ext', type=str,
                        help='Extension of ss prediction files',
                        default=".sspfa")
    parser.add_argument('--mutation_file', type=str,
                        help='Filename of mutation files. Usually "mutations.txt". Leave empty to not use mutation data')
    parser.add_argument('--params', type=str, required=True,
                    help='Pretrained parameters file')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length of inputs.')
    parser.add_argument('--preprocess', type=str, default="frequency_max_location",
                    help='Type of preprocessing for input data. Choices: nominal_location, frequency_location, frequency_max_location')
    parser.add_argument('--model', type=str, default="forest",
                    help='Name to use for this model')
    return parser.parse_args()

def preprocess(X, mut_position, max_len, pprocess):
    if mut_position is not None:
        X = mutation_nominal_data(X, mut_position)
        X = np.expand_dims(X, axis=0)
        return nominal_location_preprocess(X)
    else:
        X = nominal_data(X)
        X = np.expand_dims(X, axis=0)
        return pprocess(X, max_len)

def commands(args, predictions, mut_position=None):
    first_pred = next(iter(predictions.values()))
    preds_len = len(first_pred.seq)
    X = preprocess(predictions.values(), mut_position, args.seq_len, args.preprocess)
    with open(args.params, 'rb') as f:
        cls = pickle.load(f)
    result = ""
    q8_ss = get_ss_q8()
    for i in range(preds_len):
        y_hat = cls.predict(X[i].reshape(1, -1))
        cur_ss = q8_ss[int(y_hat.item())]
        if cur_ss == "_":
            cur_ss = "C"
        result += cur_ss
    id_split = first_pred.id.split('_')
    out_id = f"{id_split[0]}_{id_split[1]}_{args.methods}_{args.model}"
    return {out_id: result}

def main():
    args = parse_commandline()
    predictors = choose_methods(args.methods)
    args.preprocess = choose_preprocess(args.preprocess)
    work_on_predicting(args, commands, predictors)

main()