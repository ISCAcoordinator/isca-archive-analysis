import importlib.resources
import nltk

nltk.download("stopwords")


isca_stop_words = [l.strip for l in importlib.resources.read_text("isca_archive.resources", 'ISCA_stop_words.txt').split("\\n")]


# Stopwords related to data analysis
data_analysis_stopwords = {
    "analysis",
    "analytics",
    "attribute",
    "attributes",
    "data",
    "dataset",
    "datasets",
    "feature",
    "features",
    "label",
    "labels",
    "sample",
    "samples",
}

# Stopwords related to machine learning
machine_learning_stopwords = {
    "algorithm",
    "algorithms",
    "classification",
    "cluster",
    "clustering",
    "deep",
    "learn",
    "learning",
    "machine",
    "model",
    "models",
    "neural",
    "network" "predict",
    "prediction",
    "predictive",
    "regression",
    "supervised",
    "training",
    "unlabeled",
    "unsupervised",
}


def generate_stop_words(with_isca: bool = False, with_data_analysis: bool = False, with_ml: bool = False) -> set[str]:
    stopwords = set(nltk.corpus.stopwords.words("english"))

    if with_isca:
        stopwords.update(isca_stop_words)

    if with_data_analysis:
        stopwords.update(data_analysis_stopwords)

    if with_ml:
        stopwords.update(machine_learning_stopwords)

    return stopwords
