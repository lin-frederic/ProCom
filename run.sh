#!/bin/bash

# full list:
# ("imagenet" "cub" "caltech" "food" "cifarfs" "fungi" "flowers" "pets")
datasets=("flowers" "pets")
type="hierarchical"
for dataset in "${datasets[@]}"; do
    echo "Running  on $dataset"
    echo "python3 main.py --type $type --dataset $dataset -w"
    python3 main.py --type "$type" --dataset "$dataset" -w 
done
