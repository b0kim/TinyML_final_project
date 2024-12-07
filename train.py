import os
import sys
import datetime
import argparse
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput
from datasets import load_dataset

import torch_pruning as tp
from dataset import *
from evaluation import *
from utils import *

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	# general arguments
	parser.add_argument("-m", "--model", default="google/vit-base-patch16-224", help="Which model tag to load, or a path to a local .pt file")
	parser.add_argument("-d", "--dataset", required=True, choices=["imagenet-1k", "beans", "coralnet"], help="Dataset choice")
	parser.add_argument("-c", "--cache", default="~/.cache/huggingface", help="Huggingface cache directory")
	parser.add_argument("-o", "--output-root", default="./results", help="Root experiment directory")
	parser.add_argument("-en", "--experiment-name", required=True, help="What to name this experiment")
	parser.add_argument("-ev", "--no-evaluation", action="store_true", help="Whether or not to run evaluations")
	# training parameters
	parser.add_argument("-opt", "--optimizer", default="adamw_torch", type=str, choices=["sgd", "adamw_torch"], help="Optimizer choice")
	parser.add_argument("-sch", "--learning-rate-scheduler", default="linear", type=str, choices=["linear", "cosine"], help="Learning rate scheduler choice")
	parser.add_argument("-e", "--epochs", default=3, type=int, help="Number of epochs")
	parser.add_argument("-lr", "--learning-rate", default=3e-3, type=float, help="Learning rate")
	parser.add_argument("-tbs", "--train-batch-size", default=64, type=int, help="Training batch size")
	parser.add_argument("-vbs", "--val-batch-size", default=128, type=int, help="Validation batch size")
	parser.add_argument("-w", "--warmup-steps", default=0, type=int, help="Learning rate scheduler warmup steps")
	parser.add_argument("-gc", "--gradient-clipping", default=1.0, type=float, help="Gradient clipping")
	parser.add_argument("-wd", "--weight-decay", default=0.03, type=float, help="Weight decay for AdamW optimizer")
	parser.add_argument("-b1", "--beta1", default=0.9, type=float, help="beta1 parameter for AdamW optimizer")
	parser.add_argument("-b2", "--beta2", default=0.999, type=float, help="beta2 parameter for AdamW optimizer")
	args = parser.parse_args()

	# setup environment
	hprint("SETTING UP ENVIRONMENT")
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	os.environ['HF_HOME'] = args.cache
	device = "cuda" if torch.cuda.is_available() else "cpu"
	experiment_path = create_experiment_dir(args.output_root, args.experiment_name)
	write_file(" ".join(sys.argv), experiment_path, "train_command", "txt") # save command for reference
	
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
	new_model_name = f"{model_name.replace('.pt', '')}"
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
	
	# train model
	hprint(f"TRAINING MODEL")
	model.train()
	training_args = TrainingArguments(
		output_dir=f"{experiment_path}/train_{new_model_name}",
		optim=args.optimizer,
		weight_decay=args.weight_decay,
		adam_beta1=args.beta1,
		adam_beta2=args.beta2,
		lr_scheduler_type=args.learning_rate_scheduler,
		warmup_steps=args.warmup_steps,
		max_grad_norm=args.gradient_clipping,
		per_device_train_batch_size=args.train_batch_size,
		eval_strategy="steps",
		num_train_epochs=args.epochs,
		save_steps=100,
		eval_steps=100,
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
		processing_class=processor
	)
	train_results = trainer.train()

	# evaluate model
	if not args.no_evaluation:
		hprint("EVALUATING MODEL POST-TRAINING")
		val_loader = DataLoader(prepared_ds["validation"])
		footprint_metrics_postprune = evaluate_footprint(model, dummy_input, display=True)
		performance_metrics_postprune = evaluate_performance(model, val_loader, device, display=True)
		metrics_postprune = footprint_metrics_postprune | performance_metrics_postprune
		write_file(metrics_postprune, f"{experiment_path}/evaluations", new_model_name, "json")

	# save trained model
	save_model(model, f"{experiment_path}/models", new_model_name)











