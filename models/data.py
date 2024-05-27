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
    parser.add_argument('--input_preprocess', type=str,
                        help='Type of preprocessing to be done on input data. Options: nominal, onehot, frequency',
                        default="frequency")
    parser.add_argument('--target_preprocess', type=str,
                        help='Type of preprocessing to be done on target data. Options: nominal, onehot, frequency',
                        default="onehot")
    return parser.parse_args()

def onehot_encode(X):
    X_arr = np.array(list(X))
    enc = preprocessing.OneHotEncoder(categories=[list(get_ss_q8())])
    X_encoded = enc.fit_transform(X_arr[:, np.newaxis]).toarray()
    X_padded = np.pad(X_encoded, ((0,1024-len(X_encoded)),(0,0)), 'constant')
    return X_padded

def onehot_preprocess(X):
    classes = list(get_ss_q8())
    concated = np.zeros((1024*len(X),len(classes)))
    for i, prediction in enumerate(X):
        concated[i*1024:(i+1)*1024] = onehot_encode(prediction)
    return concated

def frequency_preprocess(X):
    classes = list(get_ss_q8())
    freqs = np.zeros((1024,len(classes)))
    seqs_len = len(next(iter(X)))
    for i in range(seqs_len):
        for prediction in X:
            freqs[i, ss_index(prediction[i])] += 1
    max_class_freqs = freqs.max(axis=0)
    normalized_freqs = np.divide(freqs, max_class_freqs, out=np.zeros_like(freqs), where=max_class_freqs!=0)
    return normalized_freqs

def nominal_preprocess(X):
    cats = np.zeros((1024*len(X),))
    for i, prediction in enumerate(X):
        for j in range(len(prediction)):
            cats[1024*i+j] = ss_index(prediction[j])
    return cats

def nominal_location_preprocess(X):
    nominal_X = nominal_preprocess(X)
    new_X = np.zeros(shape=(1024, len(X)*1024+1))
    for i in range(1024):
        new_X[i] = np.append(nominal_X, i)
    return new_X

#input data as mutation centered (requires mutation location knowledge)

def train_test(df, out_paths, split=0.20):
    X_train, X_test, y_train, y_test = train_test_split(df.x, df.y, test_size=split)
    write_npz(out_paths["train"], X_train, y_train)
    write_npz(out_paths["test"], X_test, y_test)
    write_npz(out_paths["all"], df.x, df.y)

def kfold(df, out_paths, split=3):
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

def choose_preprocess(type):
    if type == "nominal":
        return nominal_preprocess
    if type == "nominal_location":
        return nominal_location_preprocess
    elif type == "onehot":
        return onehot_preprocess
    elif type == "frequency":
        return frequency_preprocess
    else:
        raise ValueError("Unknown preprocessing type given in command arguments")

def main():
    args = parse_commandline()
    predictors = choose_methods(args.methods)
    input_preprocessor = choose_preprocess(args.input_preprocess)
    target_preprocessor = choose_preprocess(args.target_preprocess)
    work_on_data(args, predictors, input_preprocessor, target_preprocessor, data_process)

main()