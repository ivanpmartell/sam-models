import os
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn import preprocessing
import pandas as pd
import numpy as np

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

def read_fasta(f):
    return list(SeqIO.parse(f, "fasta"))

def get_single_record_fasta(f):
    return read_fasta(f)[0]

def keep_input_dir_structure(abs_input, abs_output, abs_directory, out_dir):
    abs_directory_no_root_folder = abs_directory[len(abs_input.rstrip('/'))+1:]
    f_out_path = os.path.join(abs_output, abs_directory_no_root_folder)
    f_out_dir = os.path.join(f_out_path, out_dir)
    return f_out_dir

def work_on_majority(args, cmds, predictors):
    abs_input_dir = os.path.abspath(args.dir)
    abs_output_dir = os.path.abspath(args.out_dir)
    if not abs_output_dir:
        abs_output_dir = abs_input_dir
    for f in Path(abs_input_dir).rglob(f"*{args.assign_ext}"):
        cluster_dir = os.path.dirname(f)
        f_basename = os.path.basename(f)
        protein = f_basename[0:f_basename.index('.')]
        out_dir = keep_input_dir_structure(abs_input_dir, abs_output_dir, cluster_dir, "majority")
        out_file = os.path.join(out_dir, f"{protein}_{args.methods}{args.pred_ext}")
        if not os.path.exists(out_file):
            predictions = dict()
            for predictor in predictors:
                prediction_path = os.path.join(cluster_dir, predictor, f"{protein}{args.pred_ext}")
                predictions[predictor] = get_single_record_fasta(prediction_path)
            assignment = get_single_record_fasta(f)
            output = cmds(args, predictions, assignment)
            write_fasta(out_file, output)

def work_on_all_data(args, predictors, preprocess, data_process):
    abs_input_dir = os.path.abspath(args.dir)
    abs_output_dir = os.path.abspath(args.out_dir)
    if not abs_output_dir:
        abs_output_dir = abs_input_dir
    out_files = {"train": os.path.join(abs_output_dir, "train_data.npz"),
                 "test": os.path.join(abs_output_dir, "test_data.npz"),
                 "all": os.path.join(abs_output_dir, "data.npz")}
    if not all([os.path.isfile(f) for f in out_files.values()]):
        training_data = []
        for f in Path(abs_input_dir).rglob(f"*{args.assign_ext}"):
            cluster_dir = os.path.dirname(f)
            f_basename = os.path.basename(f)
            protein = f_basename[0:f_basename.index('.')]
            predictions = dict()
            for predictor in predictors:
                prediction_path = os.path.join(cluster_dir, predictor, f"{protein}{args.pred_ext}")
                predictions[predictor] = get_single_record_fasta(prediction_path)
            assignment = get_single_record_fasta(f)
            x, y = preprocess(args, predictions, assignment)
            data = {"x": x, "y": y}
            training_data.append(data)
        df = pd.DataFrame.from_dict(training_data)
        data_process(args, df, out_files)

def work_on_training(args, cmds):
    abs_input = os.path.abspath(args.input)
    if not args.out_dir:
        abs_output_dir = os.path.dirname(abs_input)
    else:
        abs_output_dir = os.path.abspath(args.out_dir)
    out_file = os.path.join(abs_output_dir, f"trained.params")
    if not os.path.exists(out_file):
        X, y = load_npz(abs_input)
        cmds(args, X, y)

#sequences are a dictionary of id -> sequence (str)
def write_fasta(path, sequences):
    out_sequences = []
    for seq_id, seq in sequences.items():
        out_sequences.append(SeqRecord(Seq(seq), seq_id, "", ""))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as output_handle:
        SeqIO.write(out_sequences, output_handle, "fasta")

def write_npz(path, x, y):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, a=x, b=y)

def load_npz(path):
    loaded = np.load(path, allow_pickle=True)
    X = np.vstack(loaded["a"]).astype(float).reshape((len(loaded["a"]), 1024, 9))
    y = np.vstack(loaded["b"]).astype(float).reshape((len(loaded["b"]), 1024, 9))
    return X, y

def onehot_encode(X):
    X_arr = np.array(list(X))
    enc = preprocessing.OneHotEncoder(categories=[list(get_ss_q8())])
    X_encoded = enc.fit_transform(X_arr[:, np.newaxis]).toarray()
    X_padded = np.pad(X_encoded, ((0,1024-len(X_encoded)),(0,0)), 'constant')
    return X_padded