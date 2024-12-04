import os
import sys
import datetime
import argparse
import random
from PIL import Image
import numpy as np
from pathlib import Path
import json

import torch
from transformers import ViTForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
import datasets
from datasets import load_dataset
import evaluate
from huggingface_hub import login

import torch_pruning as tp
from footprint import evaluate_footprint

def hprint(text, symb="#"):
	print(symb*50)
	print(text)
	print(symb*50)

def process_batch(batch):
	label_key = "labels" if "labels" in batch.keys() else "label"
	inputs = processor([x for x in batch["image"]], return_tensors="pt")
	inputs["label"] = batch[label_key]
	return inputs

def collate_fn(batch):
	label_key = "labels" if "labels" in batch[0].keys() else "label"
	return {
		"pixel_values": torch.stack([x["pixel_values"] for x in batch]),
		"labels": torch.tensor([x[label_key] for x in batch])
	}

def compute_metrics(p):
	metric = evaluate.load("accuracy")
	return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	# general arguments
	parser.add_argument("-m", "--model", default="google/vit-base-patch16-224", help="Which model tag to load")
	parser.add_argument("-d", "--dataset", default="beans", choices=["imagenet-1k", "beans"], help="Which dataset to load")
	parser.add_argument("-f", "--finetune", action="store_true", help="Whether to run finetuning after pruning")
	parser.add_argument("-o", "--output", default="./results", help="Parent output directory")
	parser.add_argument("-c", "--cache", default="~/.cache/huggingface", help="Huggingface cache directory")
	parser.add_argument("-cpu", "--use-cpu", action="store_true", help="Toggle for cpu training")
	# finetuning parameters
	parser.add_argument("-opt", "--optimizer", default="sgd", help="Which optimization algorithm to use")
	parser.add_argument("-sch", "--scheduler", default="cosine", help="Learning rate scheduler")
	parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of epochs to train")
	parser.add_argument("-lr", "--learning-rate", type=float, default=3e-3, help="Learning rate for training")
	parser.add_argument("-b", "--batch-size", type=int, default=512, help="Batch size for training")
	parser.add_argument("-cl", "--clipping", type=float, default=1.0, help="Gradient clipping")
	parser.add_argument("-w", "--warmup", type=int, default=2000, help="Warmup steps")
	args = parser.parse_args()
	
	hprint("LOADING ENVIRONMENT")

	# setup environment
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	os.environ['HF_HOME'] = args.cache
	device = "cpu" if args.use_cpu else "cuda"
	login("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")
	timestamp = int(datetime.datetime.now().timestamp())
	run_path = f"{args.output}/{args.dataset}/{timestamp}"
	os.makedirs(run_path)
	
	# setup dataset
	ds = load_dataset(args.dataset, cache_dir=args.cache)
	prepared_ds = ds.with_transform(process_batch)
	label_key = "labels" if args.dataset == "beans" else "label"
	labels = ds["train"].features[label_key].names
	dummy_input = torch.randn(1, 3, 224, 224).to(device)

	# load model
	processor = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
	model = ViTForImageClassification.from_pretrained(
		args.model,
		num_labels=len(labels),
		id2label={str(i): c for i, c in enumerate(labels)},
		label2id={c: str(i) for i, c in enumerate(labels)},
		ignore_mismatched_sizes=True
	).to(device)
	hprint(model, symb="-")

	# prune model
	hprint("PRUNING MODEL")
	imp = tp.importance.RandomImportance()
	ignored_layers = [] # ignore final classification layer
	for m in model.modules():
		if isinstance(m, torch.nn.Linear) and m.out_features == len(labels):
			ignored_layers.append(m)
	pruner = tp.pruner.MetaPruner(
		model,
		dummy_input,
		importance=imp,
		pruning_ratio=0.5,
		ignored_layers=ignored_layers,
		global_pruning=True,
		isomorphic=True,
	)
	hprint("Footprint before pruning", symb="-")
	evaluate_footprint(model, dummy_input, display=True, save_path=f"{run_path}/preprune_footprint.json")
	pruner.step()
	hprint("Footprint after pruning", symb="-")
	evaluate_footprint(model, dummy_input, display=True, save_path=f"{run_path}/postprune_footprint.json")

	# setup training harness
	training_args = TrainingArguments(
		output_dir=run_path,
		optim=args.optimizer,
		use_cpu=args.use_cpu,
		lr_scheduler_type=args.scheduler,
		warmup_steps=args.warmup,
		max_grad_norm=args.clipping,
		per_device_train_batch_size=args.batch_size,
		eval_strategy="steps",
		num_train_epochs=args.epochs,
		save_steps=100,
		eval_steps=10,
		logging_steps=10,
		learning_rate=args.learning_rate,
		save_total_limit=2,
		remove_unused_columns=False,
		push_to_hub=False,
		report_to='tensorboard',
		load_best_model_at_end=True
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		data_collator=collate_fn,
		train_dataset=prepared_ds["train"],
		eval_dataset=prepared_ds["validation"],
		compute_metrics=compute_metrics,
		processing_class=processor,
	)
	
	# finetune model
	if args.finetune:
		hprint("FINETUNING MODEL")
		train_results = trainer.train()
		trainer.log_metrics("train", train_results.metrics)
		trainer.save_metrics("train", train_results.metrics)

	# evaluate model
	hprint("EVALUATING MODEL")
	eval_results = trainer.evaluate(prepared_ds["validation"])
	trainer.log_metrics("eval", eval_results)
	trainer.save_metrics("eval", eval_results)
	evaluate_footprint(model, dummy_input, display=True, save_path=f"{run_path}/postfinetune_footprint.json")

	# save results
	trainer.save_model()
	trainer.save_state()
	
	# save command
	with open(f"{run_path}/command.txt", "x") as f:
		f.write(" ".join(sys.argv))












