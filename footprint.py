import json
import torch
from torch import nn
from torchprofile import profile_macs

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def evaluate_footprint(model, inputs, display=True, save_path=None):
	metrics = dict()
	metrics["macs"] = get_model_macs(model, inputs)
	metrics["sparsity"] = get_model_sparsity(model)
	metrics["n_params"] = get_num_parameters(model)
	metrics["size"] = get_model_size(model)
	if display:
		print(f"MACs: {metrics['macs']}")
		print(f"Sparsity: {metrics['sparsity']}")
		print(f"Parameters: {metrics['n_params']/1e6} M")
		print(f"Size: {metrics['size']/(8*1000*1000)} Mb")
	if save_path is not None:
		with open(save_path, "x") as f:
			json.dump(metrics, f)

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)

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

