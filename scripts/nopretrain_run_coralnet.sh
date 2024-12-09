#!/bin/bash

# PARAMETERS
model=google/vit-base-patch16-224 # vit pretrained on imagenet-21k, finetuned on imagenet-1k
prune_name=prune_coralnet # directory name for pruning results
tune_name=tune_coralnet # directory name for finetuning results
sparsities=(0.1 0.25 0.4 0.5 0.75)
# --------------------------------------------------------------------

cd ..

# prune model
for sparsity in "${sparsities[@]}"; do
	python3 prune.py -m "$model" --dataset coralnet --experiment-name pruned_l1_"$sparsity"_coralnet --pruning-type l1 --pruning-ratio "$sparsity"
	python3 prune.py -m "$model" --dataset coralnet --experiment-name pruned_taylor_"$sparsity"_coralnet --pruning-type taylor --pruning-ratio "$sparsity"
done

mkdir ./results/"$prune_name"
mv ./results/pruned* ./results/"$prune_name" 

# finetune pruned model
for sparsity in "${sparsities[@]}"; do
	python3 train.py -m ./results/"$prune_name"/pruned_l1_"$sparsity"_coralnet/models/l1_"$sparsity"_google-vit-base-patch16-224.pt -d coralnet -lr 0.0001 -tbs 16 -vbs 16 -e 10 --experiment-name tuned_l1_"$sparsity"_coralnet
	python3 train.py -m ./results/"$prune_name"/pruned_taylor_"$sparsity"_coralnet/models/taylor_"$sparsity"_google-vit-base-patch16-224.pt -d coralnet -lr 0.0001 -tbs 16 -vbs 16 -e 10 --experiment-name tuned_taylor_"$sparsity"_coralnet
done

mkdir ./results/"$tune_name"
mv ./results/tuned* ./results/"$tune_name"