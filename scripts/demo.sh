#!/bin/bash

# PARAMETERS
model=./results/coralnet_tuned_vit/models/google-vit-base-patch16-224.pt # vit pretrained on imagenet-1k, finetuned on coralnet
prune_name=demo_prune_coralnet # directory name for pruning results
tune_name=demo_tune_coralnet # directory name for finetuning results
sparsities=(0.4)
# --------------------------------------------------------------------

cd ..

# prune model
for sparsity in "${sparsities[@]}"; do
	python3 prune.py -m "$model" --global-pruning --dataset coralnet --experiment-name pruned_taylor_"$sparsity"_coralnet --pruning-type taylor --pruning-ratio "$sparsity"
done

mkdir ./results/"$prune_name"
mv ./results/pruned* ./results/"$prune_name" 

# finetune pruned model
for sparsity in "${sparsities[@]}"; do
	python3 train.py -m ./results/"$prune_name"/pruned_taylor_"$sparsity"_coralnet/models/taylor_"$sparsity"_google-vit-base-patch16-224.pt -d coralnet -lr 0.0001 -tbs 16 -vbs 16 -e 2 --experiment-name tuned_taylor_"$sparsity"_coralnet
done

mkdir ./results/"$tune_name"
mv ./results/tuned* ./results/"$tune_name"
