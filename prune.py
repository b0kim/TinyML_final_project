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
from evaluate import *
from utils import *

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	# general arguments
	parser.add_argument("-m", "--model", default="bryanzhou008/vit-base-patch16-224-in21k-finetuned-inaturalist", help="Which model tag to load, or a path to a local .pt file")
	parser.add_argument("-d", "--dataset-root", required=True, help="Directory where dataset images are stored")
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
	
	# setup dataset
	hprint("GETTING DATALOADERS")
	if "inaturalist" in args.dataset_root:
		train_loader, val_loader, num_classes = get_inaturalist(args.dataset_root, args.train_batch_size, args.val_batch_size, 4, "phylum")
	elif "imagenet" in args.dataset_root:
		train_loader, val_loader, num_classes = get_imagenet(args.dataset_root, args.train_batch_size, args.val_batch_size, 4)
	dummy_input = torch.randn(1, 3, 224, 224).to(device)

	# load model
	hprint("LOADING MODEL")
	# TODO: load local model, and import its model_name
	model_name = args.model.replace("/", "-")
	model = ViTForImageClassification.from_pretrained(args.model, ignore_mismatched_sizes=True, num_labels=num_classes).to(device)
	new_model_name = f"{args.pruning_type}_{args.pruning_ratio}_{model_name}"
	print(model)

	# evaluate model
	if not args.no_evaluation:
		hprint("EVALUATING MODEL PRE-PRUNING")
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
		for k, (imgs, lbls) in enumerate(train_loader):
			if k>=args.taylor_batches: break
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












