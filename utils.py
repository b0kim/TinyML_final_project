import os, sys
import json

import torch

def hprint(text, symb="#"):
	print(symb*50)
	print(text)
	print(symb*50)

def create_experiment_dir(root_directory, name):
	base = f"{root_directory}/{name}"
	assert os.path.isdir(root_directory),f"Trying to create experiment directory, but root directory {root_directory} does not exist"
	assert os.path.isdir(f"{base}")==False,f"Trying to create experiment directory, but target {root_directory}/{name} already exists"
	os.makedirs(f"{base}/models")
	os.makedirs(f"{base}/evaluations")
	return base

def write_file(data, directory, name, type):
	file_path = f"{directory}/{name}.{type}"
	assert os.path.isdir(directory),f"Trying to write file, but root directory {directory} does not exist"
	assert os.path.isfile(file_path)==False,f"Trying to write file, but target {file_path} already exists"
	with open(file_path, "x") as f:
		if type == "txt":
			f.write(data)
		elif type == "json":
			json.dump(data, f)

def save_model(model, directory, name):
	file_path = f"{directory}/{name}.pt"
	assert os.path.isdir(directory),f"Trying to save model, but root directory {directory} does not exist"
	assert os.path.isfile(file_path)==False,f"Trying to save model, but target {file_path} already exists"
	model.zero_grad()
	torch.save(model, file_path)













