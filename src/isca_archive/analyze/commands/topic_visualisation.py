import pathlib
import argparse

# Bert
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset


def add_subparsers(subparsers):
    parser = subparsers.add_parser("visualize_topics", help="Visualise the topic previously extracted using BERTopic")

    parser.add_argument(
        "-i", "--inject-labels", type=str, default=None, help="Inject the custom labels (and merge topics if necessary)"
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
    embeddings = embedding_model.encode(docs["content"], show_progress_bar=True)

    # Generate the figures
    figure_dir = pathlib.Path(args.output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig = topic_model.visualize_barchart(custom_labels=True, width=500, top_n_topics=-1, n_words=-1)
    fig.write_html(figure_dir / "topics_decomposition.html")
    fig.write_image(figure_dir / "topics_decomposition.svg")

    fig = topic_model.visualize_topics(custom_labels=True)
    fig.write_html(figure_dir / "topics_map.html", include_plotlyjs=True)
    fig.write_image(figure_dir / "topics_map.svg")

    fig = topic_model.visualize_heatmap(custom_labels=True)
    fig.write_html(figure_dir / "topics_similarity_heatmap.html", include_plotlyjs=True)
    fig.write_image(figure_dir / "topics_similarity_heatmap.svg")

    fig = topic_model.visualize_documents(
        docs["content"], embeddings=embeddings, hide_document_hover=True, hide_annotations=True, custom_labels=True
    )
    fig.write_html(figure_dir / "document_embeddings.html")
    fig.write_image(figure_dir / "document_embeddings.svg")

    # with the reduced embeddings
    fig = topic_model.visualize_document_datamap(docs["content"], embeddings=embeddings)
    fig.savefig(figure_dir / "document_data_map.svg", bbox_inches="tight")

    # Distribution
    topic_distr, _ = topic_model.approximate_distribution(docs["content"])
    fig = topic_model.visualize_distribution(topic_distr[1], custom_labels=True)
    fig.write_html(figure_dir / "distribution_topics.html")
    fig.write_image(figure_dir / "distribution_topics.svg")

    # Hierarchical topic visualisation
    hierarchical_topics = topic_model.hierarchical_topics(docs["content"])
    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, custom_labels=True)
    fig.write_html(figure_dir / "hierarchical_topics.html")
    fig.write_image(figure_dir / "hierarchical_topics.svg")

    # Evolution over time
    timestamps = docs.year
    topics_over_time = topic_model.topics_over_time(docs["content"], timestamps)
    fig = topic_model.visualize_topics_over_time(topics_over_time, custom_labels=True)
    fig.write_html(figure_dir / "topics_over_time.html")
    fig.write_image(figure_dir / "topics_over_time.svg")

    # Evolution over time
    fig = topic_model.visualize_term_rank(log_scale=True, custom_labels=True)
    fig.write_html(figure_dir / "terms_rank_across_topics.html")
    fig.write_image(figure_dir / "terms_rank_across_topics.svg")
