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
    parser.add_argument('--test', type=str, required=True,
                    help='Feature selection testing methodology to be used. Options: chi2, anova, mutual_info')
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

def process(args, test_type):
    abs_input = os.path.abspath(args.input)
    X, y = load_npz(abs_input, "nominal", "nominal")
    fs = SelectKBest(score_func=test_type, k='all')
    fs.fit(X, y)
    X_train_fs = fs.transform(X)
    predictors = get_all_predictors()
    predictor_score = {}
    for i, predictor in enumerate(predictors):
        cur_score = 0
        for j in range(i*1024, (i+1)*1024):
            if not np.isnan(fs.scores_[j]):
                cur_score += fs.scores_[j]
        predictor_score[predictor] = cur_score
        print(f"{predictor}: {cur_score}")
    #Save to file
    print("Feature Selection Done!")

def main():
    args = parse_commandline()
    test_type = choose_test(args.test)
    process(args, test_type)

main()