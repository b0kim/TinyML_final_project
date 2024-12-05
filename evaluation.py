import os, sys
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
import evaluate
from evaluate import evaluator

import torch_pruning as tp

def estimate_latency(model, example_inputs, repetitions=300):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((repetitions,1))

    for _ in range(50):
        _ = model(example_inputs)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn

def compute_metrics(p):
	metric = evaluate.load("accuracy")
	return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def evaluate_performance(model, val_loader, device, display=True):
	metrics = dict()
	model.eval()
	correct = 0
	loss = 0
	with torch.no_grad():
		for batch in tqdm(val_loader):
			label_key = "labels" if "labels" in batch else "label"
			images = batch["pixel_values"]
			labels = batch[label_key]
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
	metrics["latency_mean"], metrics["latency_std"] = estimate_latency(model, inputs)
	if display:
		print(f"MACs: {metrics['macs']}")
		print(f"Sparsity: {metrics['sparsity']}")
		print(f"Parameters: {metrics['n_params']/1e6} million")
		print(f"Size: {metrics['size']/(8*1000*1000)} Mb")
		print(f"Latency: {metrics['latency_mean']} +/- {metrics['latency_std']} milliseconds")
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









