#!/bin/bash

# PARAMETERS
output_dir=ENTER_A_VALUE # output directory
model=google/vit-base-patch16-224 # huggingface pretrained model tag
dataset=ENTER_A_VALUE # local dataset path
prune_name=magnitude_prune_imagenet # directory name for pruning results
tune_name=tune_magnitude_pruned_imagenet # directory name for finetuning results

# --------------------------------------------------------------------

# prune model

# finetune pruned model
