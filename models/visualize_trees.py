import matplotlib.pyplot as plt
from sklearn import tree
import argparse
import pickle
import sys
import os

sys.path.insert(1, os.path.dirname(sys.path[0]))
from common import get_ss_q8_pred

def parse_commandline():
    parser = argparse.ArgumentParser(description='Neural network testing script')
    parser.add_argument('params', type=str,
                    help='Pretrained parameters (checkpoint) file')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where scoring metric results will be saved. Leave empty to use params directory')
    parser.add_argument('--predictors', type=int, required=True,
                    help='Amount of predictors in input data')
    parser.add_argument('--win_len', type=int, default=1024,
                    help='Length of the window to be used')
    return parser.parse_args()



def main():
    args = parse_commandline()
    with open(args.params, 'rb') as f:
        cls = pickle.load(f)
    print(len(cls.estimators_))
    fn=["af2", "colabfold", "esmfold", "sspro8"]
    cn=["0:Coil", "1:Isolated B-bridge", "2:Coil", "3:Beta sheet", "4:3-10 helix", "5:alpha helix", "6:pi helix", "7:Bend", "8:Turn"]
    for i in range(min(10,len(cls.estimators_))):
        out_file = os.path.join(args.out_dir, f'rf_tree_{i}.png')
        text_representation = tree.export_text(cls.estimators_[i],
                                               feature_names=fn,
                                               class_names=cn,
                                               show_weights=True)
        with open(f"{out_file}.log", "w") as fout:
            fout.write(text_representation)
        fig, _ = plt.subplots(nrows = 1,ncols = 1,figsize = (45,10), dpi=500)
        tree.plot_tree(cls.estimators_[i],
                    feature_names=fn,
                    class_names=cn,
                    filled = True)
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        fig.savefig(out_file)
        plt.clf()
        plt.close()

main()