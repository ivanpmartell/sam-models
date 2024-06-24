#!/bin/bash
output_processing_folder=$2
input_clusters_folder=$1
output_clusters_folder=$3
model=$4
fold=$5
preprocess=$6

python models/predict/forest.py $input_clusters_folder --methods all --params $output_processing_folder/$fold/${model}_all.params --out_dir $output_clusters_folder --model $model --preprocess $preprocess
python models/predict/forest.py $input_clusters_folder --methods top --params $output_processing_folder/$fold/${model}_top.params --out_dir $output_clusters_folder --model $model --preprocess $preprocess
python models/predict/forest.py $input_clusters_folder --methods avg --params $output_processing_folder/$fold/${model}_avg.params --out_dir $output_clusters_folder --model $model --preprocess $preprocess
python models/predict/forest.py $input_clusters_folder --methods low --params $output_processing_folder/$fold/${model}_low.params --out_dir $output_clusters_folder --model $model --preprocess $preprocess

python models/predict/majority.py $input_clusters_folder --methods all --out_dir $output_clusters_folder
python models/predict/majority.py $input_clusters_folder --methods top --out_dir $output_clusters_folder
python models/predict/majority.py $input_clusters_folder --methods avg --out_dir $output_clusters_folder
python models/predict/majority.py $input_clusters_folder --methods low --out_dir $output_clusters_folder