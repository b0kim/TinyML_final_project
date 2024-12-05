import os
import argparse

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

def get_inaturalist(inaturalist_root, train_bs=64, val_bs=128, num_workers=4, target_type="phylum"):
	"""inaturalist_root should contain train and val folders.
	"""
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		),
	])

	train_ds = torchvision.datasets.INaturalist(root=f"{inaturalist_root}/2021_train", version="2021_train", target_type=target_type, transform=transform)
	val_ds = torchvision.datasets.INaturalist(root=f"{inaturalist_root}/2021_valid", version="2021_valid", target_type=target_type, transform=transform)

	train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=num_workers)
	val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=True, num_workers=num_workers)

	# TODO: fill in values based on target_type
	if target_type == "full":
		num_classes = 10000
	elif target_type == "kingdom":
		num_classes = 3
	elif target_type == "phylum":
		num_classes = 13
	elif target_type == "class":
		raise NotImplementedError
	elif target_type == "order":
		raise NotImplementedError
	elif target_type == "family":
		raise NotImplementedError
	elif target_type == "genus":
		raise NotImplementedError

	return train_loader, val_loader, num_classes

def get_imagenet(imagenet_root, train_bs=64, val_bs=128, num_workers=4):
	"""imagenet_root should contain train and val folders.
	"""
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		),
	])

	train_ds = torchvision.datasets.ImageNet(root=f"{imagenet_root}/train", split="train", transform=transform)
	val_ds = torchvision.datasets.ImageNet(root=f"{imagenet_root}/val", split="val", transform=transform)
	
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=num_workers)
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=num_workers)
	
	num_classes = 1000

	return train_loader, val_loader, num_classes













