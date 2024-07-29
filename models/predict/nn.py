import sys
import os
import argparse
import torch

sys.path.insert(1, os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from data_preprocess import onehot_preprocess, nominal_data
from nn_models import select_model, LitModel

def parse_commandline():
    parser = argparse.ArgumentParser(description='Mutation secondary structure neural network predictor')
    parser.add_argument('dir', type=str,
                    help='Input directory containing clusters')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where trained parameters will be saved. Leave empty to use input directory')
    parser.add_argument('--model', type=str, required=True,
                    help='Type of neural network model to use')
    parser.add_argument('--params', type=str, required=True,
                    help='Pretrained parameters (checkpoint) file')
    parser.add_argument('--win_len', type=int, default=1024,
                    help='Length of the window to be used')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length')
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
    return parser.parse_args()

def preprocess(X):
    X = nominal_data(X, numtype=int)
    X = np.expand_dims(X, axis=0)
    return onehot_preprocess(X)

def predict(args, X, trained_model):
    trained_model.eval()
    with torch.no_grad():
        y_hat = []
        seq_chunk = args.win_len
        for i in range(args.seq_len // seq_chunk):
            input_cut = X[:,:,i*seq_chunk:(i+1)*seq_chunk]
            completely_masked = torch.zeros_like(input_cut)
            completely_masked[:,0,:,:] = 1
            if torch.equal(input_cut, completely_masked):
                empty_chunk = torch.zeros_like(y_hat[0])
                empty_chunk[:,0,:] = 1
                y_hat.append(empty_chunk)
            else:
                y_hat.append(trained_model(input_cut))
        return torch.cat(y_hat, 2)

def commands(args, predictions, mut_position=None):
    first_pred = next(iter(predictions.values()))
    preds_len = len(first_pred.seq)
    X = torch.Tensor(preprocess(predictions.values()))
    trained_model = LitModel.load_from_checkpoint(args.params, nnModel=args.NNModel, win_size=args.win_len, max_len=args.seq_len, predictors=args.predictors)
    pred = predict(args, X, trained_model)
    result = ""
    q8_ss = get_ss_q8()
    for i in (pred.argmax(1))[0,:preds_len]:
        result += q8_ss[i]
    id_split = first_pred.id.split('_')
    out_id = f"{id_split[0]}_{id_split[1]}_{args.methods}_{args.model}"
    return {out_id: result}

def main():
    args = parse_commandline()
    args.NNModel = select_model(args.model)
    predictors = choose_methods(args.methods)
    args.predictors = len(predictors)
    work_on_predicting(args, commands, predictors)

main()