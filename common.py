import os
from pathlib import Path
from Bio import SeqIO

def get_ss_q8():
    return "BCEGHIST"

def get_predictors():
    return ["af2", "colabfold", "esmfold", "raptorx", "rgn2", "spot1d", "spot1d_lm", "spot1d_single", "sspro8"]

def get_top_predictors():
    return ["af2", "colabfold", "esmfold", "sspro8"]

def read_fasta(f):
    return list(SeqIO.parse(f, "fasta"))

def get_single_record_fasta(f):
    return read_fasta(f)[0]

def work_on_files(args, input_dir, cmds):
    predictors = get_predictors()
    for f in Path(input_dir).rglob(f"*{args.assign_ext}"):
        cluster_dir = os.path.dirname(f)
        f_basename = os.path.basename(f)
        protein = f_basename[0:f_basename.index('.')]
        predictions = dict()
        for predictor in predictors:
            prediction_path = os.path.join(cluster_dir, predictor, f"{protein}{args.pred_ext}")
            predictions[predictor] = get_single_record_fasta(prediction_path)
        assignment = get_single_record_fasta(f)
        cmds(args, predictions, assignment)