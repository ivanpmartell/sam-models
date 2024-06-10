import os
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn import preprocessing
import pandas as pd
import numpy as np
from operator import add, sub
import pickle

def get_ss_q8():
    return "_BCEGHIST"

def ss_index(c):
    if c == 'B':
        return 1
    elif c == 'C':
        return 2
    elif c == 'E':
        return 3
    elif c == 'G':
        return 4
    elif c == 'H':
        return 5
    elif c == 'I':
        return 6
    elif c == 'S':
        return 7
    elif c == 'T':
        return 8
    else:
        raise ValueError("Wrong Q8 format")

def get_all_predictors():
    return ["af2", "colabfold", "esmfold", "raptorx", "rgn2", "spot1d", "spot1d_lm", "spot1d_single", "sspro8"]

def get_top_predictors():
    return ["af2", "colabfold", "esmfold", "sspro8"]

def get_avg_predictors():
    return ["spot1d", "spot1d_lm"]

def get_low_predictors():
    return ["raptorx", "rgn2", "spot1d_single"]

def closest_divisor(n,m):
    if n % m == 0:
        return m
    print("Split size given cannot divide the data equally.")
    for i in range(1,100):
        for f in (add,sub):
            d = f(m,i)
            if n%(d) == 0:
                print(f"Closest split size: {d}")
                return d

def choose_methods(methods):
    if methods == "all":
        return get_all_predictors()
    elif methods == "top":
        return get_top_predictors()
    elif methods == "avg":
        return get_avg_predictors()
    elif methods == "low":
        return get_low_predictors()
    else:
        split_methods = methods.split(',')
        if set(split_methods).issubset(get_all_predictors()):
            return split_methods
        else:
            raise ValueError("Unknown method or wrong method format in command arguments")

def choose_preprocess(type):
    if type == "single_target":
        return single_target_preprocess
    if type == "nominal_location":
        return nominal_location_preprocess
    elif type == "onehot":
        return onehot_preprocess
    elif type == "frequency":
        return frequency_preprocess
    elif type == "frequency_location":
        return frequency_location_preprocess
    elif type == "frequency_max_location":
        return frequency_max_location_preprocess
    else:
        raise ValueError("Unknown preprocessing type given in command arguments")

def keep_input_dir_structure(abs_input, abs_output, abs_directory, out_dir):
    abs_directory_no_root_folder = abs_directory[len(abs_input.rstrip('/'))+1:]
    f_out_path = os.path.join(abs_output, abs_directory_no_root_folder)
    f_out_dir = os.path.join(f_out_path, out_dir)
    return f_out_dir

def missing_output_is_input(args, abs_input):
    if not args.out_dir:
        abs_output_dir = abs_input
    else:
        abs_output_dir = os.path.abspath(args.out_dir)
    return abs_output_dir

def work_on_predicting(args, cmds, predictors):
    abs_input_dir = os.path.abspath(args.dir)
    abs_output_dir = missing_output_is_input(args, abs_input_dir)
    for f in Path(abs_input_dir).rglob(f"*{args.seq_ext}"):
        cluster_dir = os.path.dirname(f)
        f_basename = os.path.basename(f)
        protein = f_basename[0:f_basename.index('.')]
        out_dir = keep_input_dir_structure(abs_input_dir, abs_output_dir, cluster_dir, f"{args.model}_{args.methods}")
        out_file = os.path.join(out_dir, f"{protein}{args.pred_ext}")
        if not os.path.exists(out_file):
            predictions = dict()
            for predictor in predictors:
                prediction_path = os.path.join(cluster_dir, predictor, f"{protein}{args.pred_ext}")
                predictions[predictor] = get_single_record_fasta(prediction_path)
            #assignment = get_single_record_fasta(f)
            output = cmds(args, predictions)
            write_fasta(out_file, output)

def work_on_data(args, predictors, data_process):
    abs_input_dir = os.path.abspath(args.dir)
    abs_output_dir = missing_output_is_input(args, abs_input_dir)
    if args.split_type == "kfold":
        fname_prefix = f"{args.methods}_kfold"
        out_files = {"kfold": os.path.join(abs_output_dir, fname_prefix + "{0}.npz")}
        condition = any(fname.startswith(fname_prefix) for fname in os.listdir(abs_output_dir)) if os.path.exists(abs_output_dir) else False
    elif args.split_type == "train_test":
        out_files = {"train": os.path.join(abs_output_dir, f"{args.methods}_train.npz"),
                    "test": os.path.join(abs_output_dir, f"{args.methods}_test.npz"),
                    "all": os.path.join(abs_output_dir, f"{args.methods}_data.npz")}
        condition = all([os.path.isfile(f) for f in out_files.values()])
    if not condition:
        training_data = []
        for f in Path(abs_input_dir).rglob(f"*{args.assign_ext}"):
            #TODO:Get mutation locations from file
            cluster_dir = os.path.dirname(f)
            f_basename = os.path.basename(f)
            protein = f_basename[0:f_basename.index('.')]
            predictions = dict()
            for predictor in predictors:
                prediction_path = os.path.join(cluster_dir, predictor, f"{protein}{args.pred_ext}")
                predictions[predictor] = get_single_record_fasta(prediction_path)
            assignment = get_single_record_fasta(f)
            x = nominal_preprocess(predictions.values(), np.uint8)
            y = nominal_preprocess([assignment.seq], np.uint8)
            data = {"x": x, "y": y}
            training_data.append(data)
        df = pd.DataFrame.from_dict(training_data)
        data_process(args, df, out_files)
    else:
        print("Skipped process since output files already exist")

def work_on_training(args, cmds):
    abs_input = os.path.abspath(args.input)
    abs_output_dir = missing_output_is_input(args, os.path.dirname(abs_input))
    out_file = os.path.join(abs_output_dir, f"{args.model}_trained.params")
    if not os.path.exists(out_file):
        args.out_file = out_file
        X, y = load_npz(abs_input)
        cmds(args, X, y)
        print("Training Done!")

def work_on_testing(args, cmds):
    args.params = os.path.abspath(args.params)
    if not os.path.exists(args.params):
        print("Parameters file does not exist")
        exit(1)
    abs_input = os.path.abspath(args.input)
    abs_output_dir = missing_output_is_input(args, os.path.dirname(args.params))
    X, y = load_npz(abs_input)
    score = cmds(args, X, y)
    with open(os.path.join(abs_output_dir, f"{args.model}_score.res"), 'w') as file:
        file.write(str(score))
    print("Testing Done!")

### Data functions below
def read_fasta(f):
    return list(SeqIO.parse(f, "fasta"))

def get_single_record_fasta(f):
    return read_fasta(f)[0]

#sequences input are a dictionary of id -> sequence (str)
def write_fasta(path, sequences):
    out_sequences = []
    for seq_id, seq in sequences.items():
        out_sequences.append(SeqRecord(Seq(seq), seq_id, "", ""))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as output_handle:
        SeqIO.write(out_sequences, output_handle, "fasta")

def write_classifier(path, cls):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(cls, f)

def write_npz(path, x, y):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, a=x, b=y)

def load_npz(path):
    loaded = np.load(path, allow_pickle=True)
    X = np.vstack(loaded["a"]).astype(np.uint8)
    y = np.vstack(loaded["b"]).astype(np.uint8)
    return X, y

def onehot_encode(X):
    classes = len(get_ss_q8())
    return np.eye(classes)[X]

def onehot_preprocess(X):
    len_predictors = X.shape[-1]//1024
    classes = list(get_ss_q8())
    if len(X.shape) > 1:
        concated = np.zeros((len(X), 1024* len_predictors, len(classes)))
        for i, prediction in enumerate(X):
            concated[i] = onehot_encode(prediction)
    else:
        concated = onehot_encode(X)
    return concated

def frequency_preprocess(X):
    len_predictors = X.shape[-1]//1024
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), 1024, len_classes))
    for i in range(len(X)):
        for j in range(1024):
            for k in range(len_predictors):
                freqs[i, j, X[i,1024*k+j]] += 1
    max_class_freqs = np.full((len(X), 1024, 1), 9, dtype=np.int8)
    freqs = np.divide(freqs, max_class_freqs, out=np.zeros_like(freqs), where=max_class_freqs!=0)
    return freqs

def frequency_location_preprocess(X):
    len_predictors = X.shape[-1]//1024
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), 1024, len_classes))
    for i in range(len(X)):
        for j in range(1024):
            for k in range(len_predictors):
                freqs[i, j, X[i,1024*k+j]] += 1
    max_class_freqs = np.full((len(X), 1024, 1), 9, dtype=np.int8)
    freqs = np.divide(freqs, max_class_freqs, out=np.zeros_like(freqs), where=max_class_freqs!=0)
    freqs = freqs.swapaxes(1,2).reshape((len(X), 1024*len_classes))
    location = np.tile(np.arange(1, 1025, dtype=np.int16), len(X))
    freqs = np.repeat(freqs, repeats=1024, axis=0)
    freqs = np.c_[freqs, location]
    return freqs

def frequency_max_location_preprocess(X):
    len_predictors = X.shape[-1]//1024
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), 1024, len_classes))
    for i in range(len(X)):
        for j in range(1024):
            for k in range(len_predictors):
                freqs[i, j, X[i,1024*k+j]] += 1
    freqs = np.argmax(freqs,axis=2)
    location = np.tile(np.arange(1, 1025, dtype=np.int16), len(X))
    freqs = np.repeat(freqs, repeats=1024, axis=0)
    freqs = np.c_[freqs, location]
    return freqs

def nominal_preprocess(X, numtype=float):
    cats = np.zeros((1024*len(X),), dtype=numtype)
    for i, prediction in enumerate(X):
        for j in range(len(prediction)):
            cats[1024*i+j] = ss_index(prediction[j])
    return cats

def nominal_location_preprocess(X):
    X_len, _ = X.shape
    location = np.tile(np.arange(1, 1025, dtype=np.int16), X_len)
    X = np.repeat(X, repeats=1024, axis=0)
    X = np.c_[X, location]
    return X

def single_target_preprocess(y):
    return y.ravel().astype(float)

#input data as mutation centered (requires mutation location knowledge)