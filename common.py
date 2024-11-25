import os
import re
from pathlib import Path
from typing import NamedTuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import numpy as np
from operator import add, sub
import pickle

def get_ss_q8():
    return "_BCEGHIST"

def get_ss_q8_pred():
    return "CBCEGHIST"

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

def closest_split(n,m):
    if n % m == 0:
        return m
    print("Split size given cannot divide the data equally.")
    for i in range(1,100):
        for f in (add,sub):
            d = f(m,i)
            if n%(d) == 0:
                print(f"Closest split size: {d}")
                return d

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

def work_on_predicting(args, cmds, predictors, dir_name=""):
    abs_input_dir = os.path.abspath(args.dir)
    abs_output_dir = missing_output_is_input(args, abs_input_dir)
    if not dir_name:
        dir_name = f"{args.model}_{args.methods}"
    for f in Path(abs_input_dir).rglob(f"*{args.seq_ext}"):
        cluster_dir = os.path.dirname(f)
        f_basename = os.path.basename(f)
        print(f"Working on {f_basename}")
        protein = f_basename[0:f_basename.index('.')]
        out_dir = keep_input_dir_structure(abs_input_dir, abs_output_dir, cluster_dir, dir_name)
        out_file = os.path.join(out_dir, f"{protein}{args.pred_ext}")
        if not os.path.exists(out_file):
            predictions = dict()
            all_predictors_found = True
            missing_predictors = []
            for predictor in predictors:
                prediction_path = os.path.join(cluster_dir, predictor, f"{protein}{args.pred_ext}")
                if not os.path.exists(prediction_path):
                    all_predictors_found = False
                    missing_predictors.append(predictor)
                else:
                    predictions[predictor] = get_single_record_fasta(prediction_path)
            if not all_predictors_found:
                print(f"Skipping {f_basename}: Required method predictions missing.")
                print(missing_predictors)
                continue
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
        else:
            print("Skipping {f_basename}: Prediction already exists.")

def work_on_training(args, cmds):
    abs_input = os.path.abspath(args.input)
    abs_output_dir = missing_output_is_input(args, os.path.dirname(abs_input))
    out_file = os.path.join(abs_output_dir, f"{args.model}_trained.ckpt")
    if not os.path.exists(out_file):
        args.out_file = out_file
        X, y = load_npz(abs_input)
        cmds(args, X, y)
        print("Training Done!")
    else:
        print("Training skipped")


def remove_ckpt_ext(p: str):
    """Removes the .ckpt extension for save files

    ### Parameters
    1. p : str
        - Path of the save file

    ### Returns
    - str
        - Path without the extension
    """
    return p[:-5]

def work_on_testing(args, cmds):
    args.params = os.path.abspath(args.params)
    if not os.path.exists(args.params):
        print("Parameters (Checkpoint) file does not exist")
        exit(1)
    abs_input = os.path.abspath(args.input)
    abs_output_dir = missing_output_is_input(args, os.path.dirname(args.params))
    out_file = os.path.join(abs_output_dir, f"{args.model}_score.res")
    if not os.path.exists(out_file):
        args.out_file = out_file
        X, y = load_npz(abs_input)
        score = cmds(args, X, y)
        with open(out_file, 'w') as file:
            file.write(str(score))
        print("Testing Done!")
    else:
        print("Testing skipped")

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