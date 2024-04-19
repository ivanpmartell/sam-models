import sys
import os
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common import *

def parse_commandline():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('dir', type=str,
                    help='Directory contianing clusters')
    parser.add_argument('--pred_ext', type=str,
                        help='Extension of ss prediction files',
                        default=".sspfa")
    parser.add_argument('--assign_ext', type=str,
                        help='Extension of ss assigned files',
                        default=".ssfa")
    parser.add_argument('--switch', action='store_true',
                        help='A boolean switch')
    return parser.parse_args()

def commands(args, predictions, assignment):
    #Convert to usable input for ml models
    classes = list(get_ss_q8())
    majority = ""
    p_missmatch_count = dict.fromkeys(predictions.keys(), 0)
    for i in range(len(assignment)):
        consensus = dict.fromkeys(classes, 0)
        for predictor, prediction in predictions.items():
            if prediction[i] != assignment[i]:
                p_missmatch_count[predictor] += 1
            consensus[prediction[i]] += 1
        majority += max(consensus, key=consensus.get)
    print(assignment.id)
    missmatch_count = 0
    for i in range(len(assignment)):
        if majority[i] != assignment[i]:
            missmatch_count += 1
    print(missmatch_count)
    print(p_missmatch_count)
    #save to majority.sspfa

def main():
    args = parse_commandline()
    work_on_files(args, args.dir, commands)

main()