# Domain-specific ViT Pruning
This is a final project for MIT 6.5940, taken Fall 2024.

# Installation
The `install.sh` script will run all the installation steps described below:

1. We include a modified version of the Torch-Pruning library. To install it:

`cd Torch-Pruning && pip install -e .`

2. We include a `requirements.txt` file to help setup a python environment:

`pip install -r requirements.txt`

3. We use a subset of the CoralNet coral classification dataset: https://coralnet.ucsd.edu. To extract our sanitized dataset:

`unzip coralnet.zip`

# Usage

## Replicating our results
We provide the following scripts to run the full pruning/finetuning pipeline and replicate our results:

`./scripts/replicate_results.sh`

To run pruning and training on a single model:

`./scripts/demo.sh`

All other scripts are run in `replicate_results.sh`, and represent individual ablation experiments for different initial model configurations.

## Running individual functions
Both pruning and finetuning phases can be run individually using `prune.py` and `train.py`.

### Pruning a model
Pruning can be run using `prune.py`. The following command prunes a pre-trained ViT downloaded from Huggingface. Note that the `--cache` argument must point to the Huggingface cache directory, which by default is located at `~/.cache/huggingface`. 

`python3 prune.py --model google/vit-base-patch16-224 --cache .../.cache/huggingface --experiment-name hf_vit_magnitude_pruning`

If you want to prune custom weights that are stored locally, `--model` can also accept the path to a desired `.pt` file as input:

`python3 prune.py --model path/to/pytorch/weights.pt --cache .../.cache/huggingface --experiment-name local_vit_magnitude_pruning`

The `--pruning-type` argument can be used to select between magnitude, taylor, and randomized pruning strategies. If using taylor pruning, a calibration dataset needs to be specified:

`python3 prune.py --model google/vit-base-patch16-224 --cache .../.cache/huggingface --experiment-name taylor_pruning --pruning-type taylor --dataset coralnet`

Finally, `--global-pruning` and `--isomorphic-pruning` can be used to specify more advanced grouping strategies before pruning. By default, importance ranking occurs within each layer, and each layer is pruned to a desired sparsity. Global pruning considers all parameters in unison, while isomorphic pruning groups parameters in terms of computational topology before applying pruning:

`python3 prune.py --model google/vit-base-patch16-224 --cache .../.cache/huggingface --experiment-name global_taylor_pruning --pruning-type taylor --dataset coralnet --global-pruning`
`python3 prune.py --model google/vit-base-patch16-224 --cache .../.cache/huggingface --experiment-name isomorphic_taylor_pruning --pruning-type taylor --dataset coralnet --isomorphic-pruning`


### Finetuning a model
Finetuning can be run using `train.py`.

`python3 train.py --model google/vit-base-patch16-224 --dataset coralnet --cache .../.cache/huggingface --experiment-name training_run`

Just like with pruning, if you want to finetune custom weights that are stored locally, `--model` accepts a path to the desired `.pt` file as input:

`python3 train.py --model path/to/pytorch/weights.pt --dataset coralnet --cache .../.cache/huggingface --experiment-name training_run `

# Acknowledgements
We would like to thank the MIT 6.5940 course staff for their help throughout the semester.

We use a subset of the CoralNet open-source dataset for our experiments: https://coralnet.ucsd.edu. Our codebase was built on top of the Torch-Pruning repository: https://github.com/VainF/Torch-Pruning/tree/master. We use pre-trained models from Huggingface repositories for our experiments:

* Google ViT-B: https://huggingface.co/google/vit-base-patch16-224
* Google ViT-B-in21k: https://huggingface.co/google/vit-base-patch16-224-in21k
