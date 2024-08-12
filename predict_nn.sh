#!/bin/bash
model=$1
data_dir=$2
out_dir=$3

for i in all,9 top,4 avg,2 low,3; do
    IFS=',' read method preds <<< "${i}"
    plist=(64 128 256 512 1024)
    for j in ${plist[@]}; do
        python models/predict/nn.py $data_dir --out_dir $out_dir --params train_results/${model}_${method}_trained_${j}.ckpt --model $model --methods $method --win_len $j
    done
done
