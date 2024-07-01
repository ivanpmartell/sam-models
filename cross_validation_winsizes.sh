#!/bin/bash
output_processing_folder=$2
input_clusters_folder=$1
split=$3

plist=(0 1 2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97)
for i in ${plist[@]}; do
    echo "Working on window size $i"
    mlist=("top" "avg" "low")
    for m in ${mlist[@]}; do
        python models/cross_validation.py $input_clusters_folder --out_dir $output_processing_folder --model forest_win_$i --methods $m --split_size $split --preprocess nominal_windowed --win_side_len $i
        for ((j = 0; j <= split; j++)); do
            rm "$output_processing_folder/fold$j/forest_win_${i}_${m}.params"
        done
    done
done