#!/bin/bash
output_processing_folder=$2
input_clusters_folder=$1
split=$3

python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model Forest --methods all --split_size $split
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model Forest --methods top --split_size $split
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model Forest --methods avg --split_size $split
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model Forest --methods low --split_size $split
