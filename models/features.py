#Feature selection with sklearn chi square, anova, etc
import os
import sys
import argparse
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif


sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Random forest training script')
    parser.add_argument('input', type=str,
                    help='Input file containing data in npz format')
    parser.add_argument('--out_dir', type=str,
                    help='Output directory where feature selection results will be saved. Leave empty to use input directory')
    parser.add_argument('--test', type=str,
                    help='Feature selection testing methodology to be used. Options: chi2, anova, mutual_info')
    parser.add_argument('--params', type=str,
                    help='Parameters of classifier with feature importance scores')
    parser.add_argument('--methods', type=str,
                        help='Keyword or Comma separated list of methods to include in prediction. Keywords: all, top, avg, low',
                        default="all")
    parser.add_argument('--preprocess', type=str, default="nominal_location_preprocess",
                    help='Type of preprocessing for input data. Choices: nominal_location, frequency_location, frequency_max_location')
    parser.add_argument('--seq_len', type=int, default=1024,
                    help='Maximum sequence length of inputs.')
    parser.add_argument('--win_len', type=int, default=1024,
                    help='Window size for windowed preprocessing of inputs.')
    return parser.parse_args()

def choose_test(test):
    if test == "chi2":
        return chi2
    elif test == "anova":
        return f_classif
    elif test == "mutual_info":
        return mutual_info_classif
    else:
        raise ValueError("Unknown test type given in command arguments")

def trees_feature_importance(params_path):
    with open(params_path, 'rb') as f:
        cls = pickle.load(f)
    scores = []
    for estimator in cls.estimators_:
        scores.append(estimator.feature_importances_)
    return scores

def test_importance(in_path, test_type, max_len, preprocess):
    X, y = load_npz(in_path)
    X = preprocess(X, max_len)
    y = single_target_preprocess(y)
    fs = SelectKBest(score_func=test_type, k='all')
    fs.fit_transform(X, y)
    return [fs.scores_]

def process(out_path, predictors, scores, win_len):
    predictor_score = {}
    for predictor in predictors:
        predictor_score[predictor] = 0
    for score in scores:
        for i, predictor in enumerate(predictors):
            cur_score = 0
            for j in range(i*win_len, (i+1)*win_len):
                if not np.isnan(score[j]):
                    cur_score += score[j]
            predictor_score[predictor] += cur_score
    for predictor, cur_score in predictor_score.items():
        print(f"{predictor}: {cur_score}")
        with open(out_path, 'a') as file:
            file.write(f"{predictor}: {cur_score}\n")
    print("Feature Selection Done!")

def main():
    args = parse_commandline()
    abs_input = os.path.abspath(args.input)
    abs_output_dir = missing_output_is_input(args, os.path.dirname(abs_input))
    if args.params:
        args.test = "trees"
    else:
        if not args.test:
            print("The following arguments are required: --test")
            return 1
    out_file = os.path.join(abs_output_dir, f"{args.test}_feature_importance.res")
    if not os.path.exists(out_file):
        if args.test == "trees":
            scores = trees_feature_importance(args.params)
        else:
            test_type = choose_test(args.test)
            preprocess = choose_preprocess(args.preprocess)
            scores = test_importance(args.input, test_type, args.seq_len, preprocess)
        predictors = choose_methods(args.methods)
        process(out_file, predictors, scores, args.win_len)
    else:
        print("Feature importance output file already exists")

main()