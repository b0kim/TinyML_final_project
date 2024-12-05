import os, sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn

import torch_pruning as tp

def evaluate_performance(model, val_loader, device, display=True):
	metrics = dict()
	model.eval()
	correct = 0
	loss = 0
	with torch.no_grad():
		for images, labels in tqdm(val_loader):
			images, labels = images.to(device), labels.to(device)
			outputs = model(images).logits
			loss += F.cross_entropy(outputs, labels, reduction='sum').item()
			_, predicted = torch.max(outputs, 1)
			correct += (predicted == labels).sum().item()
	metrics["mean_accuracy"] = correct / len(val_loader.dataset)
	metrics["mean_loss"] = loss / len(val_loader.dataset)
	if display:
		print(f"Average accuracy: {metrics['mean_accuracy']}")
		print(f"Average loss: {metrics['mean_loss']}")
	return metrics

def evaluate_footprint(model, inputs, display=True):
	metrics = dict()
	metrics["macs"], metrics["n_params"] = tp.utils.count_ops_and_params(model, inputs)
	metrics["sparsity"] = get_model_sparsity(model)
	metrics["size"] = get_model_size(model)
	if display:
		print(f"MACs: {metrics['macs']}")
		print(f"Sparsity: {metrics['sparsity']}")
		print(f"Parameters: {metrics['n_params']/1e6} M")
		print(f"Size: {metrics['size']/(8*1000*1000)} Mb")
	return metrics

def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()

def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements

def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width









