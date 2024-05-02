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
    parser = argparse.ArgumentParser(description='Optional app description')
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
    parser.add_argument('--params', type=str, required=True,
                    help='Pretrained parameters file')
    return parser.parse_args()

def preprocess(predictions, preds_len):
    classes = list(get_ss_q8())
    freqs = np.zeros((1024,len(classes)))
    for i in range(preds_len):
        for prediction in predictions.values():
            freqs[i, ss_index(prediction[i])] += 1
    max_X = freqs.argmax(axis=1)
    new_X = np.zeros(shape=(1024, 1025))
    for j in range(1024):
        new_X[j] = np.append(max_X, j)
    return new_X

def commands(args, predictions):
    first_pred = next(iter(predictions.values()))
    preds_len = len(first_pred.seq)
    X = preprocess(predictions, preds_len)
    with open(args.params, 'rb') as f:
        cls = pickle.load(f)
    result = ""
    q8_ss = get_ss_q8()
    for i in range(485):
        y_hat = cls.predict(X[i].reshape(1, -1))
        result += q8_ss[int(y_hat.item())]
    id_split = first_pred.id.split('_')
    out_id = f"{id_split[0]}_{id_split[1]}_{args.methods}_{args.model}"
    return {out_id: result}

def main():
    args = parse_commandline()
    predictors = choose_methods(args.methods)
    args.model = "forest"
    work_on_predicting(args, commands, predictors)

main()