# Domain-specific ViT Pruning
This is a final project for MIT 6.5940, taken Fall 2024.

# Installation
## Datasets
We use the following two datasets to generate our results:

- Imagenet-1k: https://image-net.org/index.php
- Coralnet: https://coralnet.ucsd.edu

Imagenet is automatically downloaded through huggingface.

# Usage

## Replicating full results
We provide the following scripts to run the full pruning/finetuning pipeline and replicate our results:

'''
./scripts/run_magnitude_prune.sh
./scripts/run_taylor_prune.sh
'''

## Running individual functions
Both pruning and finetuning phases can be run individually using `prune.py` and `train.py`.

### Pruning a model
Pruning can be run using `prune.py`. The following command prunes a pre-trained ViT downloaded from Huggingface. Note that the `--cache` argument must point to the Huggingface cache directory, which by default is located at `~/.cache/huggingface`. 

'''
python3 prune.py --model google/vit-base-patch16-224 --cache .../.cache/huggingface --experiment-name vit_magnitude_pruning
'''

If you want to prune custom weights that are stored locally, `--model` can also accept the path to a desired `.pt` file as input:

'''
python3 prune.py --model path/to/pytorch/weights.pt --cache .../.cache/huggingface --experiment-name vit_magnitude_pruning
'''

The `--pruning-type` argument can be used to select between magnitude, taylor, and randomized pruning strategies. If using taylor pruning, a calibration dataset needs to be specified:

'''
python3 prune.py --model google/vit-base-patch16-224 --cache .../.cache/huggingface --experiment-name vit_magnitude_pruning --pruning-type taylor --dataset imagenet-1k
'''

### Finetuning a model
Finetuning can be run using `train.py`.

'''
python3 train.py --model google/vit-base-patch16-224 --dataset imagenet-1k --cache .../.cache/huggingface --experiment-name imagenet_training_run
'''

Just like with pruning, if you want to finetune custom weights that are stored locally, `--model` accepts a path to the desired `.pt` file as input:

'''
python3 train.py --model path/to/pytorch/weights.pt --dataset imagenet-1k --cache .../.cache/huggingface --experiment-name imagenet_training_run 
'''

# Acknowledgements
We would like to thank the MIT 6.5940 course staff. Our codebase was built on top of the Torch-Pruning repository: https://github.com/VainF/Torch-Pruning/tree/master
