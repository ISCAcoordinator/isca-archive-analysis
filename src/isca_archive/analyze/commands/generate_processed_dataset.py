import pathlib
import argparse

# Messaging/logging/progress
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

# Data processing
import pandas as pd

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessorDataset


def add_subparsers(subparsers):
	parser = subparsers.add_parser("generate_dataset", help="Generate the processed dataset")

	# Add options
	parser.add_argument("-f", "--full-text", action="store_true", help="Load full text (converted from PDF file, so it can be dirty) instead of abstract")

	# Add arguments
	parser.add_argument("isca_archive_root", help="Root directory of the ISCA archive")
	parser.add_argument("conf_ids", help="File containing the list of conference IDs to analyse")
	parser.add_argument("output_file", help="The dataframe file containing the processed ISCA Archive")

	parser.set_defaults(func=main)


def main(args: argparse.Namespace):

	# Get list conference
	list_conf: list[str] = []
	with open(args.conf_ids) as f_conf:
		list_conf = [conf.strip() for conf in f_conf.readlines()]

	# Load dataset and prepare the documents for BERTopic
	dataset = ISCAArchiveProcessorDataset(pathlib.Path(args.isca_archive_root), list_conf, load_full_text=args.full_text)
	docs = []
	with logging_redirect_tqdm():
		for i in trange(len(dataset)):
			docs.append(dataset[i])

	docs = pd.DataFrame(docs)
	docs.to_json(args.output_file, default_handler=str)
