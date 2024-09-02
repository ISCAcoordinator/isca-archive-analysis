import pathlib
import argparse

# Bert
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset
from isca_archive.analyze.common.args import parse_range


def add_subparsers(subparsers):
	parser = subparsers.add_parser("analyze_topics", help="Use BERTopic to do the topic analysis")

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
	parser.add_argument("input_dataframe", help="the ISCA Archive processed dataframe file")
	parser.add_argument("output_dir", help="The output directory")

	parser.set_defaults(func=main)


def main(args: argparse.Namespace):
	# Load the dataset
	series = args.serie_subset
	if series is not None:
		series = series.split(",")

	years = args.year_subset

	dataset = ISCAArchiveProcessedDataset(args.input_dataframe, series=series, years=years)
	docs = dataset.df
	abstracts = docs["abstract"]

	# Prepare some refinment based on https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html
	embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
	vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
	hdbscan_model = HDBSCAN(
		min_cluster_size=10,
		metric="euclidean",
		cluster_selection_method="eom",
		prediction_data=True,
	)
	umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)

	# Generate Embeddings
	embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

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
	topic_model = BERTopic(
		embedding_model=embedding_model,
		umap_model=umap_model,
		hdbscan_model=hdbscan_model,
		vectorizer_model=vectorizer_model,
		representation_model=representation_model,
		top_n_words=10,  # FIXME: hardcoded
		nr_topics=40,  # FIXME: hardcoded
		verbose=True,
	)
	# Train model
	topics, probs = topic_model.fit_transform(abstracts, embeddings)
	docs["topics"] = topics
	docs["probs"] = probs

	# TODO: check if this brings something intereting?
	keybert_topic_labels = {
		topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()
	}
	topic_model.set_topic_labels(keybert_topic_labels)

	# Reduce outliers with pre-calculate embeddings instead
	new_topics = topic_model.reduce_outliers(abstracts, topics, strategy="embeddings", embeddings=embeddings)

	# # NOTE: It is important to realize that updating the topics this
	# # way may lead to errors if topic reduction or topic merging
	# # techniques are used afterwards. The reason for this is that when
	# # you assign a -1 document to topic 1 and another -1 document to
	# # topic 2, it is unclear how you map the -1 documents. Is it
	# # matched to topic 1 or 2.
	# topic_model.update_topics(docs, topics=new_topics)

	# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
	reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine").fit_transform(embeddings)

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

	docs.to_csv(output_dir / "docs_with_topics.tsv", sep="\t", index=False)
