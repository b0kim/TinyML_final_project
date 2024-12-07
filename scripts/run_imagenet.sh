#!/bin/bash

# PARAMETERS
model=google/vit-base-patch16-224 # huggingface pretrained model tag
prune_name=prune_imagenet # directory name for pruning results
tune_name=tune_imagenet # directory name for finetuning results
sparsities=(0.1 0.25 0.4 0.5 0.75)
# --------------------------------------------------------------------

cd ..

# prune model
for sparsity in "${sparsities[@]}"; do
	python3 prune.py -m "$model" --dataset imagenet-1k --experiment-name pruned_l1_"$sparsity"_imagenet --pruning-type l1 --pruning-ratio "$sparsity"
	python3 prune.py -m "$model" --dataset imagenet-1k --experiment-name pruned_taylor_"$sparsity"_imagenet --pruning-type taylor --pruning-ratio "$sparsity"
done

mkdir ./results/"$prune_name"
mv ./results/pruned* ./results/"$prune_name" 

# finetune pruned model
for sparsity in "${sparsities[@]}"; do
	python3 train.py -m ./results/"$prune_name"/pruned_l1_"$sparsity"_imagenet/models/l1_"$sparsity"_google-vit-base-patch16-224.pt -d imagenet-1k --experiment-name tuned_l1_"$sparsity"_imagenet
	python3 train.py -m ./results/"$prune_name"/pruned_taylor_"$sparsity"_imagenet/models/taylor_"$sparsity"_google-vit-base-patch16-224.pt -d imagenet-1k --experiment-name tuned_taylor_"$sparsity"_imagenet
done

mkdir ./results/"$tune_name"
mv ./results/tuned* ./results/"$tune_name"
