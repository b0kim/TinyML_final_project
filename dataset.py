import os
import argparse

import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from datasets import load_dataset
from transformers import AutoImageProcessor

def get_num_classes(dataset):
	if dataset == "imagenet-1k":
		return 1000
	elif dataset == "beans":
		return 3
	elif dataset == "coralnet":
		return 7

def get_imagenet(cache, model_tag):
	"""cache should point to huggingface cache (default ~/.cache/huggingface)
	"""
	processor = AutoImageProcessor.from_pretrained(model_tag, use_fast=True)

	def process_batch(batch):
		inputs = processor([x for x in batch["image"]], return_tensors="pt")
		inputs["label"] = batch["label"]
		return inputs

	def collate_fn(batch):
		return {
			"pixel_values": torch.stack([x["pixel_values"] for x in batch]),
			"labels": torch.tensor([x["label"] for x in batch])
		}

	ds = load_dataset("imagenet-1k", cache_dir=cache)
	prepared_ds = ds.with_transform(process_batch)
	return prepared_ds, processor, collate_fn

def get_beans(cache, model_tag):
	"""cache should point to huggingface cache (default ~/.cache/huggingface)
	"""
	processor = AutoImageProcessor.from_pretrained(model_tag, use_fast=True)

	def process_batch(batch):
		inputs = processor([x for x in batch["image"]], return_tensors="pt")
		inputs["label"] = batch["labels"]
		return inputs

	def collate_fn(batch):
		return {
			"pixel_values": torch.stack([x["pixel_values"] for x in batch]),
			"labels": torch.tensor([x["label"] for x in batch])
		}

	ds = load_dataset("beans", cache_dir=cache)
	prepared_ds = ds.with_transform(process_batch)
	return prepared_ds, processor, collate_fn

def get_coralnet(coralnet_root, train_batch_size=64, val_batch_size=128, num_workers=4):

	base_image_path = f"{coralnet_root}/images"

	class_to_label = {"Hard coral": 0, "Soft Substrate": 1, "Hard Substrate": 2, "Algae": 3, "Seagrass": 4, "Other Invertebrates": 5, "Other": 6}

	class FullSubstrateDataset(torch.utils.data.Dataset):
	    def __init__(self, csv_path):
	        self.data = pd.read_csv(csv_path)
	        self.images = self.data.loc[:, "# patch_name"]
	        self.labels = self.data.loc[:, "high_level_label"]

	    def __len__(self):
	        return len(self.data)

	    def __getitem__(self, index):
	        label = class_to_label[self.labels[index]]
	        image_name = self.images[index]
	        img_path = os.path.join(base_image_path, image_name)
	        image = read_image(img_path)
	        return image, label

	class SubsetSubstrateDataset(torch.utils.data.Dataset):
	    def __init__(self, csv_path, indices, transform=None):
	        full_data = pd.read_csv(csv_path)
	        self.data = full_data.iloc[indices]
	        self.images = self.data.loc[:, "# patch_name"]
	        self.labels = self.data.loc[:, "high_level_label"]
	        self.transform = transform

	    def __len__(self):
	        return len(self.data)

	    def __getitem__(self, index):
	        label = class_to_label[self.labels[index]]
	        # label = class_to_label[self.labels.iloc[index]]
	        image_name = self.images[index]
	        # image_name = self.images.iloc[index]
	        img_path = os.path.join(base_image_path, image_name)
	        image = read_image(img_path)
	        if self.transform:
	            image = self.transform(image)
	        return image, label

	transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	csv_path = f"{coralnet_root}/cleaned_verified_annotations.csv"
	dataset = FullSubstrateDataset(csv_path)
	dataset = pd.read_csv(csv_path)
	val_split = 0.2
	val_size = int(np.floor(len(dataset) * val_split))
	train_size = len(dataset) - val_size

	train_ds, val_ds, _ = torch.utils.data.random_split(dataset, (train_size, val_size))
	train_set = SubsetSubstrateDataset(csv_path, train_ds.indices, transform=transform)
	val_set = SubsetSubstrateDataset(csv_path, val_ds.indices, transform=transform)

	train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=num_workers)
	val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=True, num_workers=num_workers)

	return train_loader, val_loader















