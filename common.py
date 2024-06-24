import os
import re
import math
from pathlib import Path
from typing import NamedTuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn import preprocessing
import pandas as pd
import numpy as np
from operator import add, sub
import pickle
from numpy.lib.stride_tricks import as_strided

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
    elif type == "nominal_windowed":
        return nominal_windowed_preprocess
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
            if args.mutation_file:
                mut_path = os.path.join(cluster_dir, args.mutation_file)
                all_mutations = read_mutations(mut_path)
                prot_muts = mutations_in_protein(all_mutations, protein)
                if len(prot_muts) == 0:
                    mut_position = all_mutations[0].position_
                else:
                    mut_position = prot_muts[0].position_
            else:
                mut_position = None
            output = cmds(args, predictions, mut_position)
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
            cluster_dir = os.path.dirname(f)
            f_basename = os.path.basename(f)
            protein = f_basename[0:f_basename.index('.')]
            predictions = dict()
            for predictor in predictors:
                prediction_path = os.path.join(cluster_dir, predictor, f"{protein}{args.pred_ext}")
                predictions[predictor] = get_single_record_fasta(prediction_path)
            assignment = get_single_record_fasta(f)
            if args.mutation_file:
                mut_path = os.path.join(cluster_dir, args.mutation_file)
                mutations = mutations_in_protein(read_mutations(mut_path), protein)
                if len(mutations) != 1:
                    continue
                mut_position = mutations[0].position_
                x = mutation_nominal_data(predictions.values(), mut_position, np.uint8)
                y = nominal_data([assignment.seq], np.uint8)
            else:
                x = nominal_data(predictions.values(), np.uint8)
                y = nominal_data([assignment.seq], np.uint8)
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
class Mutation(NamedTuple):
    from_: str
    position_: int
    to_: str
    proteins_: list

def read_mutations(f):
    mutations = []
    with open(f, 'r') as file:
        for mutation in file:
            pos_regex = r"(\w)(\d+)(\w) \[(.*)\]"
            from_str, pos_str, to_str, proteins = re.match(pos_regex, mutation, flags=re.IGNORECASE).groups()
            proteins_list = proteins.replace('[', '').replace(']', '').replace('"', '')
            proteins_list = proteins_list.split(',')
            position = int(pos_str)
            mut = Mutation(from_str, position, to_str, proteins_list)
            mutations.append(mut)
    return mutations

def mutations_in_protein(mutations, protein):
    muts = []
    for mut in mutations:
        if protein in mut.proteins_:
            muts.append(mut)
    return muts

def read_fasta(f):
    return list(SeqIO.parse(f, "fasta"))

def get_single_record_fasta(f):
    return read_fasta(f)[0]

def read_score(f):
    with open(f, 'r') as file:
        return float(file.readline())



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

def onehot_preprocess(Xy, max_len=1024, windowed=False):
    len_predictors = Xy.shape[-1]//max_len
    classes = list(get_ss_q8())
    if len(Xy.shape) > 1:
        concated = np.zeros((len(Xy), max_len* len_predictors, len(classes)))
        for i, prediction in enumerate(Xy):
            concated[i] = onehot_encode(prediction)
    else:
        concated = onehot_encode(Xy)
    return concated

def frequency_preprocess(X, max_len=1024, windowed=False):
    len_predictors = X.shape[-1]//max_len
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), max_len, len_classes))
    for i in range(len(X)):
        for j in range(max_len):
            for k in range(len_predictors):
                freqs[i, j, X[i,max_len*k+j]] += 1
    max_class_freqs = np.full((len(X), max_len, 1), 9, dtype=np.int8)
    freqs = np.divide(freqs, max_class_freqs, out=np.zeros_like(freqs), where=max_class_freqs!=0)
    return freqs

def frequency_location_preprocess(X, max_len=1024, circular=False):
    len_predictors = X.shape[-1]//max_len
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), max_len, len_classes))
    for i in range(len(X)):
        for j in range(max_len):
            for k in range(len_predictors):
                freqs[i, j, X[i,max_len*k+j]] += 1
    max_class_freqs = np.full((len(X), max_len, 1), 9, dtype=np.int8)
    freqs = np.divide(freqs, max_class_freqs, out=np.zeros_like(freqs), where=max_class_freqs!=0)
    freqs = freqs.swapaxes(1,2).reshape((len(X), max_len*len_classes))
    freqs = np.repeat(freqs, repeats=max_len, axis=0)
    if circular:
        for i in range(len_classes):
            freqs[:, i*max_len:i*max_len+max_len] = custom_roll(freqs[:, i*max_len:i*max_len+max_len], max_len)
    else:
        location = np.tile(np.arange(1, max_len+1, dtype=np.int16), len(X))
        freqs = np.c_[freqs, location]
    return freqs

def frequency_circular_location_preprocess(X, max_len=1024):
    return frequency_location_preprocess(X, max_len, True)

def frequency_max_preprocess(X, max_len=1024, windowed=False):
    len_predictors = X.shape[-1]//max_len
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), max_len, len_classes))
    for i in range(len(X)):
        for j in range(max_len):
            for k in range(len_predictors):
                freqs[i, j, X[i,max_len*k+j]] += 1
    freqs = np.argmax(freqs,axis=2)
    return freqs

def frequency_max_location_preprocess(X, max_len=1024, circular=False):
    len_predictors = X.shape[-1]//max_len
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), max_len, len_classes))
    for i in range(len(X)):
        for j in range(max_len):
            for k in range(len_predictors):
                freqs[i, j, X[i,max_len*k+j]] += 1
    freqs = np.argmax(freqs,axis=2)
    freqs = np.repeat(freqs, repeats=max_len, axis=0)
    if circular:
        freqs = custom_roll(freqs, max_len)
    else:
        location = np.tile(np.arange(1, max_len+1, dtype=np.int16), len(X))
        freqs = np.c_[freqs, location]
    return freqs

def frequency_circular_max_location_preprocess(X, max_len=1024):
    return frequency_max_location_preprocess(X, max_len, True)

def nominal_location_preprocess(X, max_len=1024, circular=False):
    X_len = X.shape[0]
    len_predictors = X.shape[-1]//max_len
    X = np.repeat(X, repeats=max_len, axis=0)
    if circular:
        for i in range(len_predictors):
            X[:, i*max_len:i*max_len+max_len] = custom_roll(X[:, i*max_len:i*max_len+max_len], max_len)
    else:
        location = np.tile(np.arange(1, max_len+1, dtype=np.int16), X_len)
        X = np.c_[X, location]
    return X

def nominal_circular_location_preprocess(X, max_len=1024):
    return nominal_location_preprocess(X, max_len, True)

def custom_roll(arr, max_len=1024):
    m = np.arange(max_len)*-1
    m = np.tile(m, len(arr)//max_len)
    arr_roll = arr[:, [*range(arr.shape[1]),*range(arr.shape[1]-1)]].copy()
    strd_0, strd_1 = arr_roll.strides
    n = arr.shape[1]
    result = as_strided(arr_roll, (*arr.shape, n), (strd_0 ,strd_1, strd_1))
    return result[np.arange(arr.shape[0]), (n-m)%n]

def single_target_preprocess(y):
    return y.ravel().astype(float)

def nominal_data(X, numtype=float, max_len=1024):
    cats = np.zeros((max_len*len(X),), dtype=numtype)
    for i, prediction in enumerate(X):
        for j in range(len(prediction)):
            cats[max_len*i+j] = ss_index(prediction[j])
    return cats

def mutation_nominal_data(X, mut_position, numtype=float, max_len=2048):
    cats = np.zeros((max_len*len(X),), dtype=numtype)
    for i, prediction in enumerate(X):
        for j in range(len(prediction)):
            cats[max_len*i+j] = ss_index(prediction[j])
        cats[i*max_len:i*max_len+max_len] = np.roll(cats[i*max_len:i*max_len+max_len], (max_len//2)-mut_position+1) #midway point is the centered mutation location (e.g. 1024 for 2048 max len)
    return cats

# Window side length is used for each side of the current amino acid
def nominal_windowed_preprocess(Xy, max_len, max_window_side_len=20):
    Xy_len = Xy.shape[0]
    len_predictors = Xy.shape[-1]//max_len
    window_len = max_window_side_len * 2 + 1
    Xy = Xy.reshape((Xy_len, len_predictors, max_len))
    Xy = np.pad(Xy, ((0, 0), (0, 0), (max_window_side_len, max_window_side_len)))
    res = np.zeros((Xy_len * max_len, window_len * len_predictors))
    for i in range(max_len):
        res[i::max_len] = Xy[:,:,i:window_len+i].reshape((Xy_len, -1))
    return res

def frequency_windowed_preprocess(Xy, max_window_side_len=20, max_len=1024):
    window_len = max_window_side_len * 2 + 1
    return Xy.reshape(math.ceil(max_len/window_len), window_len)

def onehot_windowed_preprocess(Xy, max_window_side_len=20, max_len=1024):
    window_len = max_window_side_len * 2 + 1
    return Xy.reshape(math.ceil(max_len/window_len), window_len)