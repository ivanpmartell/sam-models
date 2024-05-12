#!/bin/bash
fold="fold3"
output_processing_folder="/home/ivan/data_testing"
input_clusters_folder=$1
output_clusters_folder=$2

python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model Forest --methods all
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model Forest --methods top
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model Forest --methods avg
python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model Forest --methods low

python models/predict/forest.py $input_clusters_folder --methods all --params $output_processing_folder/$fold/Forest_all.params --out_dir $output_clusters_folder
python models/predict/forest.py $input_clusters_folder --methods top --params $output_processing_folder/$fold/Forest_top.params --out_dir $output_clusters_folder
python models/predict/forest.py $input_clusters_folder --methods avg --params $output_processing_folder/$fold/Forest_avg.params --out_dir $output_clusters_folder
python models/predict/forest.py $input_clusters_folder --methods low --params $output_processing_folder/$fold/Forest_low.params --out_dir $output_clusters_folder