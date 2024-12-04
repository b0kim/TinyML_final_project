import os
import argparse

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loaders(dataset_path):
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		),
	])

	train_ds = torchvision.datasets.INaturalist(root=f"{dataset_path}/2021_train", version="2021_train", transform=transform)
	valid_ds = torchvision.datasets.INaturalist(root=f"{dataset_path}/2021_valid", version="2021_valid", transform=transform)

	train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
	valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=True)

	return train_loader, valid_loader

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--directory", required=True, help="Dataset directory")
	args = parser.parse_args()

	train_loader, valid_loader = get_loaders(args.directory)
	print(train_loader)
	print(valid_loader)
	














