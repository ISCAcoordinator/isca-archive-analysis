import pathlib
import argparse
import logging

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
from kmeans_pytorch import KMeans as PTKMeans

import torch
import numpy as np
import pandas as pd

# File / Dataset
from isca_archive.analyze.common.stopwords import generate_stop_words
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset
from isca_archive.analyze.common.args import parse_range
from isca_archive.analyze.common.llama_helpers import configure_llama


def batchify(a, batch_size=512):
    n = (len(a) // batch_size) + len(a) % batch_size
    for i in np.array_split(a, n, axis=0):
        yield i


class BalancedKMeans(PTKMeans):
    def __init__(
        self,
        n_clusters=None,
        cluster_centers=None,
        device=torch.device("cpu"),
        balanced: bool = False,
        tol: float = 5e-2,
        iter_limit: int = 1000,
    ):
        super().__init__(n_clusters, cluster_centers, device, balanced)
        self._tol = tol
        self._iter_limit = iter_limit
        self.labels_ = None

    def fit(self, X, y=None):
        self.labels_ = super().fit(torch.Tensor(X), iter_limit=self._iter_limit, tol=self._tol)
        print(self.labels_.unique(return_counts=True))
        # print(y.shape)
        self.labels_ = self.labels_.numpy()
        return self

    # def transform(self, X: np.ndarray) -> np.ndarray:
    # 	return X


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
        "--ignore-isca-stopwords",
        default=False,
        action="store_true",
        help="Add the ISCA stop words to the list of stop words",
    )
    parser.add_argument(
        "--ignore-data-keywords",
        default=False,
        action="store_true",
        help="Add the data analysis keywords to the list of stop words",
    )
    parser.add_argument(
        "--ignore-ml-keywords",
        default=False,
        action="store_true",
        help="Add the Machine Learning keywords to the list of stop words",
    )

    # Control
    parser.add_argument("--nb-topics", default=14, type=int, help="The number of topics")
    parser.add_argument("--nb-words", default=20, type=int, help="The number of top words per topic")
    parser.add_argument(
        "--clustering-algorithm",
        default="kmeans",
        type=str,
        help="The clustering algorithm used (kmeans, hdbscan or kmeans_balanced)",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help='The "random" seed for a better reproducibility (both set for UMAP and Balanced KMeans)',
    )

    # LLM support
    parser.add_argument(
        "--use-llama",
        default=None,
        type=str,
        help="Activate the use of Llama (the argument corresponds to the huggingface token)",
    )

    # Add arguments
    parser.add_argument("input_dataframe", help="the ISCA Archive processed dataframe file")
    parser.add_argument("output_dir", help="The output directory")

    parser.set_defaults(func=main)


def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    # Load the dataset
    series = args.serie_subset
    if series is not None:
        series = series.split(",")

    years = args.year_subset

    dataset = ISCAArchiveProcessedDataset(args.input_dataframe, series=series, years=years)
    docs = dataset.df
    text = docs["content"]

    stop_words = generate_stop_words(args.ignore_isca_stopwords, args.ignore_data_keywords, args.ignore_ml_keywords)

    # Prepare some refinment based on https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer_model = CountVectorizer(stop_words=list(stop_words), min_df=2, ngram_range=(1, 2))

    if args.clustering_algorithm.lower() == "kmeans":
        cluster_model = KMeans(n_clusters=args.nb_topics)
    elif args.clustering_algorithm.lower() == "balanced_kmeans":
        cluster_model = BalancedKMeans(n_clusters=args.nb_topics, device=torch.device("cuda:0"), balanced=True)
    elif args.clustering_algorithm.lower() == "hdbscan":
        cluster_model = HDBSCAN(
            min_cluster_size=5,  # NOTE: hardcoded
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    else:
        raise Exception('The clustering algorithm "{args.clustering_algorithm}" is not supported')

    umap_model = UMAP(
        n_neighbors=5, n_components=10, metric="cosine", random_state=args.seed  # NOTE: hardcoded  # NOTE: hardcoded
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

    handlers = logging.getLogger("root").handlers
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
            topic: " | ".join(list(zip(*values))[0][:3])
            for topic, values in topic_model.topic_aspects_["KeyBERT"].items()
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
        # save_ctfidf=True, # FIXME: this leads to a serialization error, but I don't quite get why
        save_embedding_model=embedding_model,
    )

    docs.to_json(output_dir / "docs_with_topics.json", default_handler=str)

    # Extract custom labels => enable manual editing
    freq_df = topic_model.get_topic_freq()
    freq_df["Custom Label"] = [
        topic_model.custom_labels_[row.Topic + topic_model._outliers] for _, row in freq_df.iterrows()
    ]
    freq_df.sort_values(by="Topic", inplace=True)
    freq_df.to_csv(output_dir / "model/topic2label.tsv", sep="\t", index=False)
