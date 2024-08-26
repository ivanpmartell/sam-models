#!/bin/bash
input_clusters_folder=$1
output_clusters_folder=$2

python models/predict/majority.py $input_clusters_folder --methods all --out_dir $output_clusters_folder
python models/predict/majority.py $input_clusters_folder --methods top --out_dir $output_clusters_folder
python models/predict/majority.py $input_clusters_folder --methods avg --out_dir $output_clusters_folder
python models/predict/majority.py $input_clusters_folder --methods low --out_dir $output_clusters_folder