import pathlib
import argparse
import re

# Plotting helpers
import matplotlib.pyplot as plt

# NLP
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset
from isca_archive.analyze.common.stopwords import generate_stop_words
from isca_archive.analyze.common.args import parse_range


def add_subparsers(subparsers):
    parser = subparsers.add_parser(
        "generate_word_cloud", help="Generate the word cloud of the given set of conferences"
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
        "--ignore-isca-stopwords",
        default=False,
        action="store_true",
        help="Add the research papers/abstracts' keywords to the list of stop words",
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

    # Add arguments
    parser.add_argument("input_dataframe", help="the ISCA Archive processed dataframe file")
    parser.add_argument("output_dir", help="The output directory")

    parser.set_defaults(func=main)


def main(args: argparse.Namespace):
    # Deal with stopword
    stopwords = generate_stop_words(args.ignore_isca_stopwords, args.ignore_data_keywords, args.ignore_ml_keywords)

    # Load the dataset
    series = args.serie_subset
    if series is not None:
        series = series.split(",")

    years = args.year_subset

    dataset = ISCAArchiveProcessedDataset(args.input_dataframe, series=series, years=years)
    docs = dataset.df

    # Show wordcloud
    figure_dir = pathlib.Path(args.output_dir)
    figure_dir.mkdir(exist_ok=True, parents=True)

    for serie in series:
        for year in years:
            subset = docs.loc[(docs.serie.str.lower() == serie.lower()) & (docs.year == year)]
            if subset.empty:
                continue

            create_wordcloud(
                df=subset,
                stopwords=stopwords,
                output_file=figure_dir / f"cloud_{year}_{serie}.svg",
            )


def create_wordcloud(df, stopwords, output_file, max_words=500):
    # Generte the text
    text = " ".join(df["content"].dropna()).lower()
    text = re.sub("[^a-z -]+", "", text)

    # stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    lemma_list_of_words = []
    for word in text.split(" "):
        if word not in stopwords:
            # word = stemmer.stem(word) # NOTE: I leave it there, but it leads to non-sense outcome
            word = lemmatizer.lemmatize(word)
            if word not in stopwords:
                lemma_list_of_words.append(word)
    text = " ".join(lemma_list_of_words)

    wc_gen = WordCloud(background_color="white", max_words=max_words)
    wc = wc_gen.generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_file)
