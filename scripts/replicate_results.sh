#!/bin/bash

# pretrain google/vit-base-patch16-224
python3 train.py -m google/vit-base-patch16-224-in21k -d coralnet -lr 0.0001 -tbs 16 -vbs 16 -e 10 --experiment-name coralnet_tuned_vit

# pretrain google/vit-base-patch16-224-in21k
python3 train.py -m google/vit-base-patch16-224-in21k -d coralnet -lr 0.0001 -tbs 16 -vbs 16 -e 10 --experiment-name coralnet_tuned_vit_in21k

# pretrain runs
./pretrain_run_coralnet.sh

# no pretrain runs
./nopretrain_run_coralnet.sh

# pretrain in21k runs
./in21k_pretrain_run_coralnet.sh

# no pretrain in21k runs
./in21k_nopretrain_run_coralnet.sh


















