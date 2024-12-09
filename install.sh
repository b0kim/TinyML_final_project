#!/usr/bin/env bash

# unzip dataset
unzip coralnet.zip

# install python libraries
pip3 install -r requirements.txt

# install modified torch-pruning library
cd Torch-Pruning && pip3 install -e .



