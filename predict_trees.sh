#!/bin/bash
data_dir=$1
out_dir=$2

mlist=("decisiontree" "randomforest" "extratree")
for model in ${mlist[@]}; do
    methodlist=("sspro8,colabfold" "sspro8,esmfold" "sspro8,colabfold,esmfold")
    for method in ${methodlist[@]}; do
        plist=(0 1 2 3 5)
        for j in ${plist[@]}; do
            echo "Working on ${model}_win_${j}_${method}"
            python models/predict/trees.py $data_dir --out_dir $out_dir --params tree_results/fold0/${model}_win_${j}_${method}.ckpt --model $model --methods $method --preprocess nominal_windowed --win_side_len $j
        done
    done
done