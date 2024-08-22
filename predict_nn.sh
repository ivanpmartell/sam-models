#!/bin/bash
data_dir=$1
out_dir=$2

mlist=("fullyconnected" "convolutional" "recurrent" "transformer")
for model in ${mlist[@]}; do
    methodlist=("sspro8,colabfold" "sspro8,esmfold" "sspro8,colabfold,esmfold")
    for method in ${methodlist[@]}; do
        plist=(64 128 256 512 1024)
        for j in ${plist[@]}; do
            python models/predict/nn.py $data_dir --out_dir $out_dir --params nn_results/${model}_${method}_trained_${j}.ckpt --model $model --methods $method --win_len $j
        done
    done
done