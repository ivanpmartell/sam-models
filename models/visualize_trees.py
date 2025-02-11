import matplotlib.pyplot as plt
from sklearn import tree
import argparse
import pickle
import sys
import os

sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import get_ss_q8_pred

def parse_commandline():
    parser = argparse.ArgumentParser(description='Tree visualizations')
    parser.add_argument('params', type=str,
                    help='Pretrained parameters (checkpoint) file')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where scoring metric results will be saved. Leave empty to use params directory')
    parser.add_argument('--predictors', type=str, required=True,
                    help='Comma-separated list of predictor names in same order as when trained')
    parser.add_argument('--model', type=str, default="extratree",
                    help='Type of tree model to use. Choices: ExtraTree, RandomForest, DecisionTree')
    parser.add_argument('--images', action="store_true", 
                    help='Create tree plots as images. Warning: Outputs very big images and is a slow process')
    return parser.parse_args()

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
        out_file = os.path.join(args.out_dir, f'{args.model}_{i}.png')
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        text_representation = tree.export_text(cls.estimators_[i],
                                               feature_names=fn,
                                               class_names=cn,
                                               show_weights=True,
                                               max_depth=50,
                                               decimals=0)
        with open(f"{out_file}.log", "w") as fout:
            fout.write(text_representation)
        if args.images:
            fig, _ = plt.subplots(nrows = 1,ncols = 1,figsize = (45,10), dpi=500)
            tree.plot_tree(cls.estimators_[i],
                        feature_names=fn,
                        class_names=cn,
                        filled = True)
            fig.savefig(out_file)
            plt.clf()
            plt.close()

main()