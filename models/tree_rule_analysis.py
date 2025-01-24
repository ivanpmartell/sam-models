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
    return parser.parse_args()

def pretty_print(dic):
    sorted_keys = sorted(dic.keys())
    for key in sorted_keys:
        print(f"\"{key}\": {np.array2string(dic[key], separator=', ')}")

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
    pretty_print(class_weights)


main()