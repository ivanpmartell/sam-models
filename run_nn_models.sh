#!/bin/bash
outDir=$1

mlist=("fullyconnected" "convolutional" "recurrent" "transformer")
for model in ${mlist[@]}; do
    for i in sspro8,colabfold-2 sspro8,esmfold-2 sspro8,colabfold,esmfold-3; do
        IFS='-' read method preds <<< "${i}"
        plist=(64 128 256 512 1024)
        for j in ${plist[@]}; do
            python models/data.py ~/clusters/ --out_dir nominal_data --methods $method
            python models/train/nn.py nominal_data/${method}_train.npz --out_dir $outDir --model $model --predictors $preds --win_len $j
            mv $outDir/${model}_trained.ckpt $outDir/${model}_${method}_trained_${j}.ckpt
            python models/test/nn.py nominal_data/${method}_test.npz --out_dir $outDir --params $outDir/${model}_${method}_trained_${j}.ckpt --model $model --predictors $preds --win_len $j
            mv $outDir/${model}_score.res $outDir/${model}_${method}_score_${j}.res
        done
    done
done