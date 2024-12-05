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
	# training parameters
	parser.add_argument("-e", "--epochs", default=3, type=int, help="Number of epochs")
	parser.add_argument("-lr", "--learning-rate", default=3e-3, type=float, help="Learning rate")
	parser.add_argument("-tbs", "--train-batch-size", default=64, type=int, help="Training batch size")
	parser.add_argument("-vbs", "--val-batch-size", default=128, type=int, help="Validation batch size")
	args = parser.parse_args()

	# setup environment
	hprint("SETTING UP ENVIRONMENT")
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	device = "cuda" if torch.cuda.is_available() else "cpu"
	experiment_path = create_experiment_dir(args.output_root, args.experiment_name)
	write_file(" ".join(sys.argv), experiment_path, "train_command", "txt") # save command for reference
	
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
	new_model_name = f"trained_{model_name}"
	print(model)

	# evaluate model
	if not args.no_evaluation:
		hprint("EVALUATING MODEL PRE-PRUNING")
		footprint_metrics_preprune = evaluate_footprint(model, dummy_input, display=True)
		performance_metrics_preprune = evaluate_performance(model, val_loader, device, display=True)
		metrics_preprune = footprint_metrics_preprune | performance_metrics_preprune
		write_file(metrics_preprune, f"{experiment_path}/evaluations", model_name, "json")
	
	# train model
	hprint(f"TRAINING MODEL")
	model.train()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss()
	for epoch in range(args.epochs):
		total_loss = 0
		total_correct = 0
		for images, labels in tqdm(train_loader):
			images, labels = images.to(device), labels.to(device)
			logits = model(images).logits
			loss = criterion(logits, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			_, predicted = torch.max(logits, 1)
			correct = (predicted == labels).sum().item()
			total_correct += correct
			print(f"loss: {loss.item()}, accuracy: {correct/args.train_batch_size}")
		hprint(f"Epoch {epoch}: loss {total_loss/(epoch*len(train_loader))}, accuracy {total_correct/(epoch*len(train_loader))}", symb="-")

	# evaluate model
	if not args.no_evaluation:
		hprint("EVALUATING MODEL POST-PRUNING")
		footprint_metrics_postprune = evaluate_footprint(model, dummy_input, display=True)
		performance_metrics_postprune = evaluate_performance(model, val_loader, device, display=True)
		metrics_postprune = footprint_metrics_postprune | performance_metrics_postprune
		write_file(metrics_postprune, f"{experiment_path}/evaluations", new_model_name, "json")

	# save trained model
	save_model(model, f"{experiment_path}/models", new_model_name)











