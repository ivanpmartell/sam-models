import os
import sys
import argparse
import subprocess

sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Random forest training script')
    parser.add_argument('input', type=str,
                    help='Input folder with the cluster data')
    parser.add_argument('--out_dir', type=str, required=True,
                    help='Output folder where temporary data and results will be saved')
    parser.add_argument('--model', type=str, required=True,
                    help='Machine learning model to be used')
    parser.add_argument('--methods', type=str,
                        help='Keyword or Comma separated list of methods to include in majority consensus prediction. Keywords: all, top, avg, low',
                        default="all")
    parser.add_argument('--split_size', type=str,
                        help='Amount of data to split. Values:1 to 100',
                        default="5")
    parser.add_argument('--preprocess', type=str, default="frequency_max_location",
                    help='Type of preprocessing for input data. Choices: nominal_location, frequency_location, frequency_max_location')
    parser.add_argument('--win_side_len', type=int, default=1024,
                    help='Window size for windowed preprocessing of inputs.')
    return parser.parse_args()

def select_model(model):
    model = model.lower()
    if "randomforest" or "decisiontree" or "extratree" in model:
        return "{0}/forest.py"
    elif "fullyconnected" in model:
        return "{0}/nn.py"
    elif "convolutional" in model:
        return "{0}/nn.py"
    elif "recurrent" in model:
        return "{0}/nn.py"
    elif "transformer" in model:
        return "{0}/nn.py"
    else:
        print("Incorrect model, please choose: RandomForest, DecisionTree, ExtraTree, FullyConnected, Convolutional, Recurrent, Transformer")
        exit(1)

def pipeline(args):
    # Run data for kfold
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    abs_out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(abs_out_dir, "data")
    subprocess.check_call(['python', os.path.join(current_path, 'data.py'), args.input, '--out_dir', data_dir, '--split_type', 'kfold', '--split_size', args.split_size, '--methods', args.methods])
    # loop for train and test (leave one out) for each fold
    fold_accs = []
    fold_list = list(range(int(args.split_size)))
    for i in fold_list:
        fold_dir = os.path.join(abs_out_dir, f"fold{i}")
        train_data_path = os.path.join(fold_dir, f"{args.methods}_train.npz")
        if not os.path.isfile(train_data_path):
            train_folds = fold_list[:i] + fold_list[i+1 :]
            train_data = [load_npz(os.path.join(data_dir, f"{args.methods}_kfold{train_fold}.npz")) for train_fold in train_folds]
            X_train = train_data[0][0]
            y_train = train_data[0][1]
            for j in range(1, len(train_data)):
                X_train = np.concatenate([X_train, train_data[j][0]])
                y_train = np.concatenate([y_train, train_data[j][1]])
            write_npz(train_data_path, X_train, y_train)
        test_data_path = os.path.join(fold_dir, f"{args.methods}_test.npz")
        if not os.path.isfile(test_data_path):
            test_fold = fold_list[i]
            X_test, y_test = load_npz(os.path.join(data_dir, f"{args.methods}_kfold{test_fold}.npz"))
            write_npz(test_data_path, X_test, y_test)
        # Training
        script_path = os.path.join(current_path, select_model(args.model))
        subprocess.call(['python', script_path.format("train"), train_data_path, '--out_dir', fold_dir, "--model", args.model, "--preprocess", args.preprocess, "--win_side_len", str(args.win_side_len)])
        # Testing
        params_path = os.path.join(fold_dir, f"{args.model}_trained.ckpt")
        subprocess.call(['python', script_path.format("test"), test_data_path, '--params', params_path, "--model", args.model, "--preprocess", args.preprocess, "--win_side_len", str(args.win_side_len)])
        os.rename(params_path, os.path.join(fold_dir, f"{args.model}_{args.methods}.ckpt"))
        scores_path = os.path.join(fold_dir, f"{args.model}_score.res")
        fold_accs.append(read_score(scores_path))
        os.rename(scores_path, os.path.join(fold_dir, f"{args.model}_{args.methods}.score"))
    #print and save file with agglomerated results of complete cross-validation
    print(f"Cross validation mean accuracy: {sum(fold_accs)/len(fold_accs)}")


def main():
    args = parse_commandline()
    pipeline(args)

main()