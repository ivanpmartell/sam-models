import sys
import os
import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from nn_models import select_model, select_device

def parse_commandline():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('dir', type=str,
                    help='Input directory containing clusters')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    parser.add_argument('--model', type=str, required=True,
                    help='Type of neural network model to use')
    parser.add_argument('--params', type=str, required=True,
                    help='Pretrained parameters file')
    parser.add_argument('--methods', type=str,
                        help='Keyword or Comma separated list of methods to include in majority consensus prediction. Keywords: all, top, avg, low',
                        default="all")
    parser.add_argument('--seq_ext', type=str,
                        help='Extension of ss assigned files',
                        default=".fa")
    parser.add_argument('--pred_ext', type=str,
                        help='Extension of ss prediction files',
                        default=".sspfa")
    return parser.parse_args()

def preprocess(predictions, preds_len):
    classes = list(get_ss_q8())
    freqs = np.zeros((1024,len(classes)))
    for i in range(preds_len):
        for prediction in predictions.values():
            freqs[i, ss_index(prediction[i])] += 1
    max_class_freqs = freqs.max(axis=0)
    normalized_freqs = np.divide(freqs, max_class_freqs, out=np.zeros_like(freqs), where=max_class_freqs!=0)
    return normalized_freqs

def commands(args, predictions):
    first_pred = next(iter(predictions.values()))
    preds_len = len(first_pred.seq)
    X = torch.Tensor(preprocess(predictions, preds_len))
    X.unsqueeze_(0)
    model = args.NNModel().to(args.device)
    model.load_state_dict(torch.load(args.params))
    model.eval()
    result = ""
    ss_q8 = get_ss_q8()
    with torch.no_grad():
        pred = model(X)
        for i in (pred.argmax(2))[0,:preds_len]:
            result += ss_q8[i]
    id_split = first_pred.id.split('_')
    out_id = f"{id_split[0]}_{id_split[1]}_{args.methods}_{args.model}"
    return {out_id: result}

def main():
    args = parse_commandline()
    args.model = args.model.lower()
    args.NNModel = select_model(args.model)
    args.device = select_device()
    predictors = choose_methods(args.methods)
    work_on_predicting(args, commands, predictors)

main()