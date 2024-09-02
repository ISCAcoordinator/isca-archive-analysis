import pathlib
import argparse
from loguru import logger

# Bert
from sentence_transformers import SentenceTransformer
from bertopic.representation import ZeroShotClassification
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# Data
import pandas as pd

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset
from isca_archive.analyze.common.args import parse_range

def add_subparsers(subparsers):
	parser = subparsers.add_parser("zero_shot_topic", help="Apply zero shot topic analysis")

	parser.add_argument(
		"-S",
		"--secondary-areas",
		action="store_true",
		help="Use the secondary areas instead of the primary ones",
	)

	parser.add_argument(
		"-s",
		"--serie-subset",
		default=None,
		help="Comma separated of the series to focus on",
	)
	parser.add_argument(
		"-y",
		"--year-subset",
		default=None,
		type=parse_range,
		help="Comma separated of the years to focus on",
	)

	parser.add_argument(
		"--ignore-research-keywords",
		default=False,
		action="store_true",
		help="Add the research papers/abstracts' keywords to the list of stop words"
	)
	parser.add_argument(
		"--ignore-data-keywords",
		default=False,
		action="store_true",
		help="Add the data analysis keywords to the list of stop words"
	)
	parser.add_argument(
		"--ignore-ml-keywords",
		default=False,
		action="store_true",
		help="Add the Machine Learning keywords to the list of stop words"
	)

	# Add arguments
	parser.add_argument(
		"input_dataframe", help="the ISCA Archive processed dataframe file"
	)
	parser.add_argument(
		"area_dataframe", help="The dataframe file containg the IS areas"
	)
	parser.add_argument("output_dir", help="The output directory")

	parser.set_defaults(func=main)


def main(args: argparse.Namespace):

	# Load candidate topics
	df_topics = pd.read_csv(args.area_dataframe, sep="\t")
	if args.secondary_areas:
		logger.info("Use the secondary areas")
		df_topics = df_topics[~pd.isna(df_topics.Secondary)]
	else:
		logger.info("Use the primary areas")
		df_topics = df_topics[pd.isna(df_topics.Secondary)]
	candidate_topics = list(df_topics.Description)

	# Load the dataset
	years = args.year_subset
	series = args.serie_subset
	if series is not None:
		series = series.split(",")

	dataset = ISCAArchiveProcessedDataset(
		args.input_dataframe, series=series, years=years
	)
	docs = dataset.df
	abstracts = docs.reset_index()["abstract"]

	# # Create your representation model
	# representation_model = ZeroShotClassification(candidate_topics, model="facebook/bart-large-mnli")

	# Use the representation model in BERTopic on top of the default pipeline
	topic_model = BERTopic(
		embedding_model="thenlper/gte-small",
		min_topic_size=15,
		zeroshot_topic_list=candidate_topics,
		zeroshot_min_similarity=0.85,
		representation_model=KeyBERTInspired()
	)
	topics, _ = topic_model.fit_transform(abstracts)
	docs["topics"] = topics
	_, probs = topic_model.transform(abstracts)
	docs["probs"] = probs

	topic_info = topic_model.get_topic_info()

	##########################################################################################

	# Save model & updated dataframe
	output_dir = pathlib.Path(args.output_dir)
	if not output_dir.exists():
		output_dir.mkdir(parents=True)

	model_dir = output_dir / "model"
	model_dir.mkdir(parents=True, exist_ok=True)
	topic_model.save(
		model_dir,
		serialization="safetensors",
	)

	docs.to_csv(output_dir / "docs_with_topics.tsv", sep="\t", index_label="AbstractID")
	topic_info.to_csv(output_dir / "topic_info.tsv", sep="\t", index=False)
