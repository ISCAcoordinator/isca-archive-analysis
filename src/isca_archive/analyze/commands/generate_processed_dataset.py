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
	parser.add_argument("-k", "--isca-keywords", action="store_true", help="Only load the ISCA Archive keywords extracted during the publication phase (incompatible with full text)")
	parser.add_argument("-t", "--use-title", action="store_true", help="Add the title as part of the data to be loaded")

	# Add arguments
	parser.add_argument("isca_archive_root", help="Root directory of the ISCA archive")
	parser.add_argument("conf_ids", help="File containing the list of conference IDs to analyse")
	parser.add_argument("output_file", help="The dataframe file containing the processed ISCA Archive")

	parser.set_defaults(func=main)


def main(args: argparse.Namespace):
	if args.full_text and args.isca_keywords:
		raise Exception("full text and keywords extraction are mutually exclusive and can't be used at the same time!")

	# Get list conference
	list_conf: list[str] = []
	with open(args.conf_ids) as f_conf:
		list_conf = [conf.strip() for conf in f_conf.readlines()]

	# Load dataset and prepare the documents for BERTopic
	dataset = ISCAArchiveProcessorDataset(pathlib.Path(args.isca_archive_root), list_conf, load_full_text=args.full_text, use_isca_keywords=args.isca_keywords, use_title=args.use_title)
	docs = []
	with logging_redirect_tqdm():
		for i in trange(len(dataset)):
			docs.append(dataset[i])

	docs = pd.DataFrame(docs)

	# NOTE: post process to be sure a document doesn't contain any abstract
	if (args.isca_keywords or args.full_text):
		docs = docs[docs.content_type != "abstract"]

	# NOTE: let's exclude rows without ISCA (FIXME: this is hardcoded!)
	docs = docs[~pd.isna(docs.author_area_id)]
	docs.to_json(args.output_file, default_handler=str)
