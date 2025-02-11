import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import _tree
import argparse
import pickle
import sys
import os
import numpy as np
from copy import deepcopy

sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import get_ss_q8_pred

def parse_commandline():
    parser = argparse.ArgumentParser(description='Obtain readable decisions from a trained tree-type model')
    parser.add_argument('params', type=str,
                    help='Pretrained parameters (checkpoint) file')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where scoring metric results will be saved. Leave empty to use params directory')
    parser.add_argument('--predictors', type=str, required=True,
                    help='Predictor names in same order as when trained')
    parser.add_argument('--model', type=str, default="extratree",
                    help='Type of tree model to use. Choices: ExtraTree, RandomForest, DecisionTree')
    return parser.parse_args()

def tree_to_code(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    decision_dict = {n : set(class_names) for n in feature_names}
    def recurse(node, depth, decisions):
        txt = ""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            classes = [c for idx, c in enumerate(class_names) if idx <= threshold]
            left_dec = deepcopy(decisions)
            left_dec[name] = left_dec[name].intersection(classes)
            txt += recurse(tree_.children_left[node], depth + 1, left_dec)
            classes = [c for idx, c in enumerate(class_names) if idx > threshold]
            right_dec = decisions
            right_dec[name] = right_dec[name].intersection(classes)
            txt += recurse(tree_.children_right[node], depth + 1, right_dec)
        else:
            output = class_names[np.argmax(tree_.value[node])]
            conditions = ""
            for predictor, sse in decisions.items():
                conditions += f"{predictor} is {sse}; "
            conditions = conditions[:-2]
            txt = f"Predicted {output} when {conditions}\n"
        return txt

    return recurse(0, 1, decision_dict)

def main():
    args = parse_commandline()
    args.params = os.path.abspath(args.params)
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.params)
    with open(args.params, 'rb') as f:
        cls = pickle.load(f)
    if "decisiontree" in args.params.lower():
        cls.estimators_ = [cls]
    fn = args.predictors.split(",")
    cn=["0:Coil", "1:Isolated B-bridge", "2:Coil", "3:Beta sheet", "4:3-10 helix", "5:alpha helix", "6:pi helix", "7:Bend", "8:Turn"]
    for i in range(min(10,len(cls.estimators_))):
        out_file = os.path.join(args.out_dir, f'{args.model}_{i}.dcn')
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        text_decisions = tree_to_code(cls.estimators_[i], feature_names=fn, class_names=cn)
        with open(out_file, "w") as fout:
            fout.write(text_decisions)

main()