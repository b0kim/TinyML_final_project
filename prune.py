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

import footprint

def hprint(text):
	print('-'*50)
	print(text)
	print('-'*50)

def process_batch(batch):
	if "labels" in batch.keys():
		label_key = "labels"
	else:
		label_key = "label"
	inputs = processor([x for x in batch["image"]], return_tensors="pt")
	inputs["label"] = batch[label_key]
	return inputs

def collate_fn(batch):
	if "labels" in batch[0].keys():
		label_key = "labels"
	else:
		label_key = "label"
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
	device = "cuda"
	if args.use_cpu:
		device = "cpu"
	login("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")
	timestamp = int(datetime.datetime.now().timestamp())
	
	# setup dataset
	ds = load_dataset(args.dataset, cache_dir=args.cache)
	prepared_ds = ds.with_transform(process_batch)
	if args.dataset == "beans":
		label_key = "labels"
	else:
		label_key = "label"
	labels = ds["train"].features[label_key].names
	
	# load model
	processor = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
	model = ViTForImageClassification.from_pretrained(
		args.model,
		num_labels=len(labels),
		id2label={str(i): c for i, c in enumerate(labels)},
		label2id={c: str(i) for i, c in enumerate(labels)},
		ignore_mismatched_sizes=True
	)
	
	# setup training harness
	training_args = TrainingArguments(
		output_dir=f"{args.output}/{args.dataset}/{timestamp}",
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
	footprint_eval = dict()
	footprint_eval["model_size"] = footprint.get_model_size(model)
	footprint_eval["model_sparsity"] = footprint.get_model_sparsity(model)
	footprint_eval["model_macs"]= footprint.get_model_macs(model, torch.randn(1, 3, 224, 224).to(device))
	footprint_eval["model_num_parameters"] = footprint.get_num_parameters(model)
	print("MODEL SIZE (Mb): ", footprint_eval["model_size"]/(8*1000*1000))
	print("SPARSITY: ", footprint_eval["model_sparsity"])
	print("MACS: ", footprint_eval["model_macs"])
	print("NUM PARAMS: ", footprint_eval["model_num_parameters"])
	with open(f"{args.output}/{args.dataset}/{timestamp}/footprint_eval.json", "w") as f:
		json.dump(footprint_eval, f)

	# save results
	trainer.save_model()
	trainer.save_state()
	
	# save command
	with open(f"{args.output}/{args.dataset}/{timestamp}/command.txt", "x") as f:
		f.write(" ".join(sys.argv))












