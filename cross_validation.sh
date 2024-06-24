#!/bin/bash
output_processing_folder=$2
input_clusters_folder=$1
split=$3
model=$4
preprocess=$5

python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model $model --methods all --split_size $split --preprocess $preprocess
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model $model --methods top --split_size $split --preprocess $preprocess
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model $model --methods avg --split_size $split --preprocess $preprocess
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model $model --methods low --split_size $split --preprocess $preprocess