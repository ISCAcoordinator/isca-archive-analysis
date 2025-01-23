import pathlib
import argparse

# BERTopic
from bertopic import BERTopic
from bertopic.representation import BaseRepresentation
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans


import pandas as pd

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset
from isca_archive.analyze.common.args import parse_range


def add_subparsers(subparsers):
	parser = subparsers.add_parser("analyze_topics", help="Use BERTopic to do the topic analysis")

	# Subsetting options
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

	# Ignoring options
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

	# Control
	parser.add_argument(
		"--nb-topics",
		default=40,
		type=int,
		help="The number of topics"
	)
	parser.add_argument(
		"--nb-words",
		default=20,
		type=int,
		help="The number of top words per topic"
	)

	# LLM support
	parser.add_argument(
		"--use-llama",
		default=None,
		type=str,
		help="Activate the use of Llama (the argument corresponds to the huggingface token)"
	)

	# Add arguments
	parser.add_argument("input_dataframe", help="the ISCA Archive processed dataframe file")
	parser.add_argument("output_dir", help="The output directory")

	parser.set_defaults(func=main)


def configure_llama(token: str) -> BaseRepresentation:
	# Some local imports
	from bertopic.representation import TextGeneration
	import huggingface_hub
	import transformers
	from torch import cuda
	from torch import bfloat16

	# Some local constants
	MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'
	# MODEL_ID = 'meta-llama/Meta-Llama-3-70B-Instruct'
	# MODEL_ID = 'meta-llama/Llama-3.1-8B'
	DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

	# Login to huggingface hub
	huggingface_hub.login(token=token)

	bnb_config = transformers.BitsAndBytesConfig(
		load_in_4bit=True,  # 4-bit quantization
		bnb_4bit_quant_type='nf4',  # Normalized float 4
		bnb_4bit_use_double_quant=True,  # Second quantization after the first
		bnb_4bit_compute_dtype=bfloat16  # Computation type
	)


	# Prepare Llama 2
	tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
	model = transformers.AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		trust_remote_code=True,
		quantization_config=bnb_config,
		device_map='auto',
	)
	model.eval()


	# Our text generator
	generator = transformers.pipeline(
		model=model, tokenizer=tokenizer,
		task='text-generation',
		temperature=0.1,
		max_new_tokens=500,
		repetition_penalty=1.1
	)

	# System prompt describes information given to all conversations
	system_prompt = """
	<s>[INST] <<SYS>>
	You are a helpful, respectful and honest expert in speech science and speech technology acting as an assistantfor labeling topics.
	<</SYS>>
	"""

	# Example prompt demonstrating the output we are looking for
	example_prompt = """
	I have a topic that contains the following documents:
	- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
	- Meat, but especially beef, is the word food in terms of emissions.
	- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

	The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

	Based on the information about the topic above, please create a short label of this topic.
	Make sure you to only return the label and nothing more.

	[/INST] Environmental impacts of eating meat
	"""

	# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
	main_prompt = """
	[INST]
	I have a topic that contains the following documents:
	[DOCUMENTS]

	The topic is described by the following keywords: '[KEYWORDS]'.

	Based on the information about the topic above, please create a short label of this topic.
	Make sure you to only return the label and nothing more.
	[/INST]
	"""
	prompt = system_prompt + example_prompt + main_prompt

	return TextGeneration(generator, prompt=prompt)


def main(args: argparse.Namespace):
	# Load the dataset
	series = args.serie_subset
	if series is not None:
		series = series.split(",")

	years = args.year_subset

	dataset = ISCAArchiveProcessedDataset(args.input_dataframe, series=series, years=years)
	docs = dataset.df
	text = docs["content"]

	# Prepare some refinment based on https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html
	embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
	vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

	cluster_model = KMeans(n_clusters=50)
	# cluster_model = HDBSCAN(
	# 	min_cluster_size=5, # NOTE: hardcoded
	# 	metric="euclidean",
	# 	cluster_selection_method="eom",
	# 	prediction_data=True,
	# )
	umap_model = UMAP(
		n_neighbors=5, # NOTE: hardcoded
		n_components=10, # NOTE: hardcoded
		metric="cosine",
	)

	# Generate Embeddings
	embeddings = embedding_model.encode(text, show_progress_bar=True)

	# Play with three types of representation
	keybert_model = KeyBERTInspired()
	pos_model = PartOfSpeech("en_core_web_sm")
	mmr_model = MaximalMarginalRelevance(diversity=0.3)

	# Add all models together to be run in a single `fit`
	representation_model = {
		"KeyBERT": keybert_model,
		# "OpenAI": openai_model,  # Uncomment if you will use OpenAI
		"MMR": mmr_model,
		"POS": pos_model,
	}

	if args.use_llama is not None:
		representation_model["Llama"] = configure_llama(args.use_llama)

	topic_model = BERTopic(
		embedding_model=embedding_model,
		umap_model=umap_model,
		hdbscan_model=cluster_model,
		vectorizer_model=vectorizer_model,
		representation_model=representation_model,
		top_n_words=args.nb_words,
		nr_topics=args.nb_topics,
		verbose=True,
	)

	# Train model
	topics, probs = topic_model.fit_transform(text, embeddings)
	# topics = topic_model.reduce_outliers(text, topics)
	# topic_model.update_topics(text, topics=topics)
	docs["BERTopic"] = topics
	docs["probs"] = probs

	if args.use_llama is not None:
		llama_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["Llama"].values()]
		topic_model.set_topic_labels(llama_labels)
		topic_model.get_topic(1, full=True)
	else:
		# TODO: check if this brings something intereting?
		keybert_topic_labels = {
			topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()
		}
		topic_model.set_topic_labels(keybert_topic_labels)

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
		save_ctfidf=True,
		save_embedding_model=embedding_model,
	)

	docs.to_json(output_dir / "docs_with_topics.json", default_handler=str)

	# Extract custom labels => enable manual editing
	freq_df = topic_model.get_topic_freq()
	freq_df["Custom Label"] = [topic_model.custom_labels_[row.Topic+topic_model._outliers] for _, row in freq_df.iterrows()]
	freq_df.sort_values(by="Topic", inplace=True)
	freq_df.to_csv(output_dir/"model/topic2label.tsv", sep="\t", index=False)
