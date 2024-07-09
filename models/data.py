import sys
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *
from data_preprocess import *

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
    parser.add_argument('--mutation_file', type=str,
                        help='Filename of mutation files. Usually "mutations.txt". Leave empty to not use mutation data')
    parser.add_argument('--methods', type=str,
                        help='Keyword or Comma separated list of methods to include in prediction. Keywords: all, top, avg, low',
                        default="all")
    parser.add_argument('--split_type', type=str,
                        help='Method of splitting the data. Keywords: kfold, train_test',
                        default="train_test")
    parser.add_argument('--split_size', type=str,
                        help='Amount of data to split. Values: kfold=1 to 100, train_test=0 to 1',
                        default="0.20")
    return parser.parse_args()

def train_test(df, out_paths, split):
    X_train, X_test, y_train, y_test = train_test_split(df.x, df.y, test_size=split)
    write_npz(out_paths["train"], X_train, y_train)
    write_npz(out_paths["test"], X_test, y_test)
    write_npz(out_paths["all"], df.x, df.y)

def kfold(df, out_paths, split):
    data_len = len(df.x)
    usable_split = closest_split(data_len, split)
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

def main():
    args = parse_commandline()
    predictors = choose_methods(args.methods)
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

main()