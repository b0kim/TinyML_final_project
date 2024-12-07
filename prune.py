import os
import sys
import datetime
import argparse
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import ViTForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput

import torch_pruning as tp
from dataset import *
from evaluation import *
from utils import *

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	# general arguments
	parser.add_argument("-m", "--model", default="google/vit-base-patch16-224", help="Which model tag to load, or a path to a local .pt file")
	parser.add_argument("-d", "--dataset", default="imagenet-1k", choices=["imagenet-1k", "beans", "coralnet"], help="Dataset choice")
	parser.add_argument("-c", "--cache", default="~/.cache/huggingface", help="Huggingface cache directory")
	parser.add_argument("-o", "--output-root", default="./results", help="Root experiment directory")
	parser.add_argument("-en", "--experiment-name", required=True, help="What to name this experiment")
	parser.add_argument("-ev", "--no-evaluation", action="store_true", help="Whether or not to run evaluations")
	# pruning parameters
	parser.add_argument("-tb", "--taylor-batches", default=10, type=int, help="Number of batches for taylor criterion")
	parser.add_argument("-pr", "--pruning-ratio", default=0.5, type=float, help="Pruneing ratio")
	parser.add_argument("-b", "--bottleneck", default=False, action="store_true", help="Bottleneck or uniform")
	parser.add_argument("-pt", "--pruning-type", default="l1", type=str, choices=["random", "taylor", "l1"], help="Pruning type")
	parser.add_argument("-gp", "--global-pruning", default=False, action="store_true", help="Global pruning")
	parser.add_argument("-tbs", "--train-batch-size", default=64, type=int, help="Training batch size")
	parser.add_argument("-vbs", "--val-batch-size", default=128, type=int, help="Validation batch size")
	args = parser.parse_args()

	# setup environment
	hprint("SETTING UP ENVIRONMENT")
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	device = "cuda" if torch.cuda.is_available() else "cpu"
	experiment_path = create_experiment_dir(args.output_root, args.experiment_name)
	write_file(" ".join(sys.argv), experiment_path, "prune_command", "txt") # save command for reference
	
	# load model
	hprint("LOADING MODEL")
	if os.path.isfile(args.model): # load from local file
		model_name = args.model.split("/")[-1]
		hf_model_name = args.model.split("/")[-1][:-3].split("_")[-1].replace("-", "/", 1)
		model = torch.load(args.model).to(device)
	else: # load from huggingface hub
		model_name = args.model.replace("/", "-")
		hf_model_name = args.model
		model = ViTForImageClassification.from_pretrained(hf_model_name, ignore_mismatched_sizes=True, num_labels=get_num_classes(args.dataset)).to(device)
	new_model_name = f"{args.pruning_type}_{args.pruning_ratio}_{model_name.replace('.pt', '')}"
	print(model)

	# load dataset
	hprint("LOADING DATASETS")
	if args.dataset == "imagenet-1k":
		prepared_ds, processor, collate_fn = get_imagenet(args.cache, hf_model_name)
	elif args.dataset == "beans":
		prepared_ds, processor, collate_fn = get_beans(args.cache, hf_model_name)
	elif args.dataset == "coralnet":
		prepared_ds, processor, collate_fn = get_coralnet(args.cache, hf_model_name)
	dummy_input = torch.randn(1, 3, 224, 224).to(device)

	# evaluate model
	if not args.no_evaluation:
		hprint("EVALUATING MODEL PRE-PRUNING")
		val_loader = DataLoader(prepared_ds["validation"])
		footprint_metrics_preprune = evaluate_footprint(model, dummy_input, display=True)
		performance_metrics_preprune = evaluate_performance(model, val_loader, device, display=True)
		metrics_preprune = footprint_metrics_preprune | performance_metrics_preprune
		write_file(metrics_preprune, f"{experiment_path}/evaluations", model_name, "json")
	
	# prune model
	hprint(f"PRUNING MODEL: {args.pruning_type}, {args.pruning_ratio} sparsity")
	if args.pruning_type == 'random':
		imp = tp.importance.RandomImportance()
	elif args.pruning_type == 'taylor':
		imp = tp.importance.TaylorImportance()
	elif args.pruning_type == 'l1':
		imp = tp.importance.MagnitudeImportance(p=1)
	else: raise NotImplementedError

	num_heads = {}
	ignored_layers = [model.classifier]
	for m in model.modules(): # all heads should be pruned simultaneously, so channels are grouped by head
		if isinstance(m, ViTSelfAttention):
			num_heads[m.query] = m.num_attention_heads
			num_heads[m.key] = m.num_attention_heads
			num_heads[m.value] = m.num_attention_heads
		if args.bottleneck and isinstance(m, ViTSelfOutput):
			ignored_layers.append(m.dense)
	pruner = tp.pruner.MetaPruner(
		model,
		dummy_input,
		importance=imp,
		pruning_ratio=args.pruning_ratio,
		ignored_layers=ignored_layers,
		global_pruning=args.global_pruning,
		output_transform=lambda out: out.logits.sum(),
		num_heads=num_heads,
		prune_head_dims=True,
		prune_num_heads=False,
		head_pruning_ratio=0.5,
	)
	if isinstance(imp, tp.importance.TaylorImportance):
		model.zero_grad()
		train_loader = DataLoader(prepared_ds["train"])
		for k, batch in enumerate(train_loader):
			if k>=args.taylor_batches: break
			label_key = "labels" if "labels" in batch else "label"
			imgs = batch["pixel_values"]
			lbls = batch[label_key]
			imgs = imgs.to(device)
			lbls = lbls.to(device)
			output = model(imgs).logits
			loss = torch.nn.functional.cross_entropy(output, lbls)
			loss.backward()

	for g in pruner.step(interactive=True):
		g.prune()

	for m in model.modules(): # modify the attention head size and all head sizes after pruning
		if isinstance(m, ViTSelfAttention):
			print(m)
			print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
			m.num_attention_heads = pruner.num_heads[m.query]
			m.attention_head_size = m.query.out_features // m.num_attention_heads
			m.all_head_size = m.query.out_features
			print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)
			print()

	# evaluate model
	if not args.no_evaluation:
		hprint("EVALUATING MODEL POST-PRUNING")
		footprint_metrics_postprune = evaluate_footprint(model, dummy_input, display=True)
		performance_metrics_postprune = evaluate_performance(model, val_loader, device, display=True)
		metrics_postprune = footprint_metrics_postprune | performance_metrics_postprune
		write_file(metrics_postprune, f"{experiment_path}/evaluations", new_model_name, "json")

	# save pruned model
	save_model(model, f"{experiment_path}/models", new_model_name)












