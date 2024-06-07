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

def test_importance(in_path, test_type):
    X, y = load_npz(in_path)
    X = nominal_location_preprocess(X)
    y = single_target_preprocess(y)
    fs = SelectKBest(score_func=test_type, k='all')
    fs.fit_transform(X, y)
    return [fs.scores_]

def process(out_path, predictors, scores):
    predictor_score = {}
    for predictor in predictors:
        predictor_score[predictor] = 0
    for score in scores:
        for i, predictor in enumerate(predictors):
            cur_score = 0
            for j in range(i*1024, (i+1)*1024):
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
    if args.params:
        args.test = "trees"
        scores = trees_feature_importance(args.params)
    else:
        if not args.test:
            print("The following arguments are required: --test")
            return 1
        test_type = choose_test(args.test)
        scores = test_importance(args.input, test_type)
    abs_input = os.path.abspath(args.input)
    abs_output_dir = missing_output_is_input(args, os.path.dirname(abs_input))
    out_file = os.path.join(abs_output_dir, f"{args.test}_feature_importance.res")
    predictors = choose_methods(args.methods)
    process(out_file, predictors, scores)

main()