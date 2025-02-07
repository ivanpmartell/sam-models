import argparse
import sys
import os
import numpy as np
from pathlib import Path

def parse_commandline():
    parser = argparse.ArgumentParser(description='Tree rules analysis script')
    parser.add_argument('rules_dir', type=str,
                    help='Folder containing Tree rule (text) files')
    parser.add_argument('--ext', type=str, default="log",
                    help='Extension of the text files')
    parser.add_argument('--percentage', action="store_true", 
                    help='Get results as percentage values')
    return parser.parse_args()

def pretty_print(dic, percent):
    if percent:
        np.set_printoptions(formatter={'float_kind':lambda x: "%.1f" % x + "%"})
    sorted_keys = sorted(dic.keys())
    for key in sorted_keys:
        print(f"\"{key}\": {np.array2string(dic[key], separator=', ', precision=1, floatmode='fixed', suppress_small=True)}")

def main():
    args = parse_commandline()
    current_file = 0
    class_weights = dict()
    directory = Path(os.path.abspath(args.rules_dir))
    for filepath in directory.glob(f"*.{args.ext}"):
        with open(filepath, 'r') as f:
            current_file += 1
            for line in f:
                index = line.find('weights: [')
                if index == -1:
                    continue
                line_with_weights = line[index + 10:]
                weights_str, class_str = line_with_weights.split("] class: ")
                weights_list = weights_str.split(", ")
                weights = np.array(weights_list, dtype=np.uint64)
                class_str = class_str.rstrip()
                if class_weights.get(class_str) is not None:
                    class_weights[class_str] += weights
                else:
                    class_weights[class_str] = weights
    for key in class_weights:
        class_weights[key] = np.uint64(class_weights[key] / current_file)
        if args.percentage:
            class_sum = np.sum(class_weights[key])
            class_weights[key] = np.float64(class_weights[key]) / class_sum * 100
    pretty_print(class_weights, args.percentage)


main()