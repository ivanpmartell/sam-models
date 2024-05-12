#!/bin/bash
output_processing_folder=$2
input_clusters_folder=$1
output_clusters_folder=$3
fold=$4

python models/predict/forest.py $input_clusters_folder --methods all --params $output_processing_folder/$fold/Forest_all.params --out_dir $output_clusters_folder
python models/predict/forest.py $input_clusters_folder --methods top --params $output_processing_folder/$fold/Forest_top.params --out_dir $output_clusters_folder
python models/predict/forest.py $input_clusters_folder --methods avg --params $output_processing_folder/$fold/Forest_avg.params --out_dir $output_clusters_folder
python models/predict/forest.py $input_clusters_folder --methods low --params $output_processing_folder/$fold/Forest_low.params --out_dir $output_clusters_folder