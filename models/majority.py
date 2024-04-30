import sys
import os
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('dir', type=str,
                    help='Input directory containing clusters')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where clusters and their majority prediction results will be saved. Leave empty to use input directory')
    parser.add_argument('--pred_ext', type=str,
                        help='Extension of ss prediction files',
                        default=".sspfa")
    parser.add_argument('--assign_ext', type=str,
                        help='Extension of ss assigned files',
                        default=".ssfa")
    parser.add_argument('--methods', type=str,
                        help='Keyword or Comma separated list of methods to include in majority consensus prediction. Keywords: all, top, avg, low',
                        default="all")
    return parser.parse_args()

def commands(args, predictions, assignment):
    classes = list(get_ss_q8())
    majority = ""
    for i in range(len(assignment)):
        consensus = dict.fromkeys(classes, 0)
        for prediction in predictions.values():
            consensus[prediction[i]] += 1
        majority += max(consensus, key=consensus.get)
    id_split = assignment.id.split('_')
    out_id = f"{id_split[0]}_{id_split[1]}_{args.methods}_majority"
    return {out_id: majority}


def main():
    args = parse_commandline()
    predictors = choose_methods(args.methods)
    work_on_majority(args, commands, predictors)

main()