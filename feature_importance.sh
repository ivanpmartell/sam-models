#!/bin/bash
data_folder=$1
model=$2

methods_list=(
all
top
avg
low
)
for methods in "${methods_list[@]}"; do
    python models/features.py $data_folder/${methods}_data.npz --test chi2 --methods $methods --preprocess nominal_windowed --win_len 41
    python models/features.py $data_folder/${methods}_data.npz --test anova --methods $methods --preprocess nominal_windowed --win_len 41
    python models/features.py $data_folder/${methods}_data.npz --test mutual_info --methods $methods --preprocess nominal_windowed --win_len 41
    python models/features.py test.npz --params $data_folder/${model}_${methods}.params --methods $methods --win_len 41
done