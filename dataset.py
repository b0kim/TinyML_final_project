import os
import argparse

import numpy as np
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

def get_coralnet(cache, model_tag):
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

	ds = load_dataset("imagefolder", data_dir="./images_10000")
	prepared_ds = ds.with_transform(process_batch)
	return prepared_ds, processor, collate_fn













