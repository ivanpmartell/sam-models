#!/bin/bash
input_clusters_folder=$1
output_processing_folder=$2
split=$3

delete="F"
plist=(0 1 2 3 5)
for i in ${plist[@]}; do
    echo "Working on window size $i"
    nlist=("decisiontree" "randomforest" "extratree")
    for n in ${nlist[@]}; do
        mlist=("sspro8,esmfold" "sspro8,colabfold" "sspro8,colabfold,esmfold")
        for m in ${mlist[@]}; do
            missing_file="F"
            for ((j = 0; j <= split-1; j++)); do
                if [ ! -f "$output_processing_folder/fold$j/${n}_win_${i}_${m}.score" ]; then
                    echo "$output_processing_folder/fold$j/${n}_win_${i}_${m}.score"
                    missing_file="T"
                fi
            done
            if [[ $missing_file == "T" ]]; then
                python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model ${n}_win_$i --methods $m --split_size $split --preprocess nominal_windowed --win_side_len $i
                if [[ $delete == "T" ]]; then
                    for ((j = 0; j <= split; j++)); do
                        rm "$output_processing_folder/fold$j/${n}_win_${i}_${m}.ckpt"
                    done
                fi
            fi
        done
    done
done