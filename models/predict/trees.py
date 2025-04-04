import sys
import os
import argparse
import numpy as np
import pickle

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from data_preprocess import nominal_data, mutation_nominal_data, nominal_location_preprocess, choose_preprocess

def parse_commandline():
    parser = argparse.ArgumentParser(description='Mutation secondary structure tree-type predictor')
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
                    help='Pretrained parameters (checkpoint) file')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length of inputs.')
    parser.add_argument('--preprocess', type=str, default="frequency_max_location",
                    help='Type of preprocessing for input data. Choices: nominal_location, frequency_location, frequency_max_location')
    parser.add_argument('--model', type=str, default="forest",
                    help='Name to use for this model')
    parser.add_argument('--win_side_len', type=int,
                    help='Size of window for input.')
    return parser.parse_args()

def preprocess(X, mut_position, max_len, pprocess, win_side_len):
    if mut_position is not None:
        X = mutation_nominal_data(X, mut_position)
        X = np.expand_dims(X, axis=0)
        return nominal_location_preprocess(X)
    else:
        X = nominal_data(X, numtype=int)
        X = np.expand_dims(X, axis=0)
        if win_side_len is not None:
            X = pprocess(X, max_len, win_side_len)
        else:
            X = pprocess(X, max_len)
        return X

def commands(args, predictions, mut_position=None):
    first_pred = next(iter(predictions.values()))
    preds_len = len(first_pred.seq)
    X = preprocess(predictions.values(), mut_position, args.seq_len, args.preprocess, args.win_side_len)
    with open(args.params, 'rb') as f:
        cls = pickle.load(f)
    result = ""
    q8_ss = get_ss_q8_pred()
    for i in range(preds_len):
        y_hat = cls.predict(X[i].reshape(1, -1))
        result += q8_ss[int(y_hat.item())]
    id_split = first_pred.id.split('_')
    out_id = f"{id_split[0]}_{id_split[1]}_{args.methods}_{args.model}"
    return {out_id: result}

def main():
    args = parse_commandline()
    predictors = choose_methods(args.methods)
    args.model = args.model.lower()
    args.preprocess = choose_preprocess(args.preprocess)
    work_on_predicting(args, commands, predictors, dir_name=f"{args.model}_{args.methods}_{1+(args.win_side_len*2)}")

main()