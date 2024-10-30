import os
from pathlib import Path
import argparse
from common import missing_output_is_input, write_fasta

def parse_commandline():
    parser = argparse.ArgumentParser(description='Neural network training script')
    parser.add_argument('input', type=str,
                    help='Input directory containing data in all format')
    parser.add_argument('--out_dir', type=str,
                    help='Base directory where files will be converted to fasta and ssfa format. Leave empty to use input directory')
    return parser.parse_args()

def is_ordered(csv_line):
    elements = csv_line.rstrip(',').split(',')
    elements = [range(int(elements[0]), int(elements[-1]))]
    return elements == sorted(elements)

def main():
    args = parse_commandline()
    abs_input_dir = os.path.abspath(args.input)
    abs_output_dir = missing_output_is_input(args, abs_input_dir)
    for f in Path(abs_input_dir).rglob(f"*.all"):
        f_basename = os.path.basename(f)
        f_noext = f_basename[0:f_basename.index('.')]
        fa_out_file = os.path.join(abs_output_dir, f"{f_noext}.fa")
        ssfa_out_file = os.path.join(abs_output_dir, f"{f_noext}.ssfa")
        if not (os.path.exists(fa_out_file) and os.path.exists(ssfa_out_file)):
            with open(f, 'r') as in_file:
                for line in in_file:
                    line = line.rstrip()
                    if line.startswith("RES:"):
                        seq = line[4:]
                    if line.startswith("DSSP:"):
                        ss_data = line[5:]
                    if line.startswith("RsNo:"):
                        data_num = line[5:]
            if not is_ordered(data_num):
                print(f"ERROR: residues are not ordered in {f}")
                break
            seq = seq.replace(",", "")
            ss_data = ss_data.replace(",", "").replace("_", "C")
            if len(seq) == len(ss_data):
                write_fasta(fa_out_file, {f_noext: seq})
                write_fasta(ssfa_out_file, {f_noext: ss_data})

main()