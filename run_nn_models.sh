#!/bin/bash
model=$1
outDir=$2

for i in all,9 top,4 avg,2 low,3; do
    IFS=',' read method preds <<< "${i}"
    plist=(64 128 256 512 1024)
    for j in ${plist[@]}; do
        python models/data.py ~/clusters/ --out_dir nominal_data --methods $method
        python models/train/nn.py nominal_data/${method}_train.npz --out_dir $outDir --model $model --predictors $preds --win_len $j
        mv $outDir/${model}_trained.ckpt $outDir/${model}_${method}_trained_${j}.ckpt
        python models/test/nn.py nominal_data/${method}_test.npz --out_dir $outDir --params $outDir/${model}_${method}_trained_${j}.ckpt --model $model --predictors $preds --win_len $j
        mv $outDir/${model}_score.res $outDir/${model}_${method}_score_${j}.res
    done
done