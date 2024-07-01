#!/bin/bash
data_folder=$1
model=$2

python models/features.py $data_folder/all_data.npz --test chi2 --methods all --preprocess nominal_windowed --win_len 41
python models/features.py $data_folder/all_data.npz --test anova --methods all --preprocess nominal_windowed --win_len 41
python models/features.py $data_folder/all_data.npz --test mutual_info --methods all --preprocess nominal_windowed --win_len 41
python models/features.py test.npz --params $data_folder/${model}_all.params --methods all --win_len 41