#!/bin/bash
model=$1

for i in all,9 top,4 avg,2 low,3; do
    IFS=',' read method preds <<< "${i}"
    plist=(128 256 512 1024)
    for j in ${plist[@]}; do
        python models/data.py ~/clusters/ --out_dir nominal_data --methods $method
        python models/train/nn.py nominal_data/${method}_train.npz --out_dir train_results --model $model --predictors $preds --win_len $j
        mv train_results/${model}_trained.ckpt train_results/${model}_trained_${j}.ckpt
        python models/test/nn.py nominal_data/${method}_test.npz --out_dir train_results --params train_results/${model}_trained_${j}.ckpt --model $model --predictors $preds --win_len $j
        mv train_results/${model}_score.res train_results/${model}_score_${j}.res
    done
done
