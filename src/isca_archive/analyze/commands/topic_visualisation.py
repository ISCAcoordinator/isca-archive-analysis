import pathlib
import argparse

# Bert
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset


def add_subparsers(subparsers):

	parser = subparsers.add_parser(
		"visualize_topics", help="Visualise the topic previously extracted using BERTopic"
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
	parser.add_argument("model_dir", help="The output directory")
	parser.add_argument("output_dir", help="The output directory")

	parser.set_defaults(func=main)


def main(args: argparse.Namespace):
	# Load the dataset
	dataset = ISCAArchiveProcessedDataset(
		args.input_dataframe,
	)
	docs = dataset.df

	# Load model and add embedding model
	embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
	topic_model = BERTopic.load(args.model_dir, embedding_model=embedding_model)

	# Generate Embeddings
	embeddings = embedding_model.encode(docs["abstract"], show_progress_bar=True)

	####################################################################
	# Visualisation
	figure_dir = pathlib.Path(args.output_dir)
	figure_dir.mkdir(parents=True, exist_ok=True)

	fig = topic_model.visualize_barchart()
	fig.write_html(figure_dir / "barchar.html")

	fig = topic_model.visualize_topics()
	fig.write_html(figure_dir / "topics.html", include_plotlyjs=True)

	fig = topic_model.visualize_heatmap()
	fig.write_html(figure_dir / "heatmap.html", include_plotlyjs=True)

	fig = topic_model.visualize_documents(
		docs["abstract"],
		embeddings=embeddings,
		hide_document_hover=True,
		hide_annotations=True,
	)
	fig.write_html(figure_dir / "document_embeddings.html")

	# Distribution
	topic_distr, _ = topic_model.approximate_distribution(docs["abstract"])
	fig = topic_model.visualize_distribution(topic_distr[1])
	fig.write_html(figure_dir / "topic_distribution.html")

	# Hierarchical topic visualisation
	hierarchical_topics = topic_model.hierarchical_topics(docs["abstract"])
	fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
	fig.write_html(figure_dir / "hierarchical_topics.html")

	# Evolution over time
	timestamps = docs.year
	topics_over_time = topic_model.topics_over_time(docs["abstract"], timestamps)
	fig = topic_model.visualize_topics_over_time(topics_over_time)
	fig.write_html(figure_dir / "topics_over_time.html")
