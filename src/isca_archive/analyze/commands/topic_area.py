import pathlib
import argparse

import pandas as pd
import plotly.express as px

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset


def add_subparsers(subparsers):
    parser = subparsers.add_parser(
        "topic_area",
        help="Match the ISCA area to the BERTopic and compute some useful information",
    )

    parser.add_argument(
        "-i",
        "--inject-labels",
        type=str,
        default=None,
        help="Inject the custom labels (and merge BERTopic if necessary)",
    )

    # Add arguments
    parser.add_argument("input_dataframe", help="the ISCA Archive processed dataframe file")
    # parser.add_argument("model_dir", help="The output directory")
    parser.add_argument("output_dir", help="The output directory")

    parser.set_defaults(func=main)


def main(args: argparse.Namespace):
    # Load and filter the dataset
    dataset = ISCAArchiveProcessedDataset(
        args.input_dataframe,
    )
    docs = dataset.df
    docs = docs[(docs.year > 2020)]  # NOTE: hardcoded
    docs = docs[["author_area_id", "author_area_label", "BERTopic", "title"]]
    # NOTE: nan/empty cell is weird, this should be fixed in the dataset file
    docs = docs[
        (docs.author_area_id != "")
        & (docs.author_area_id.notna())
        & (docs.author_area_id != "nan")
        & (docs.author_area_id != "Show and Tell")
    ]  # filter out documents without ISCA area
    docs.author_area_id = docs.author_area_id.apply(lambda x: int(x))
    area_labels = (
        docs[["author_area_id", "author_area_label"]]
        .drop_duplicates()
        .sort_values(by="author_area_id")["author_area_label"]
    )
    docs = docs[["author_area_id", "BERTopic", "title"]]

    dist = docs.groupby(["author_area_id", "BERTopic"]).agg("count").reset_index()
    dist.rename({"title": "count"}, axis="columns", inplace=True)
    dist = dist.pivot(index="author_area_id", columns="BERTopic", values="count")
    dist.fillna(0, inplace=True)

    # Save everything
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Plot heatmap
    fig = px.imshow(
        dist,
        labels=dict(x="BERTopic", y="Author ISCA Topic", color="Count"),
        x=dist.columns,
        y=dist.index,
        text_auto=True,  # Automatically display values in cells
    )

    fig.update_layout(
        title="Heatmap of Counts",
        xaxis_title="BERTopic",
        yaxis_title="Author ISCA Topic",
    )
    fig.update_xaxes(
        tickmode="linear",
    )
    print(area_labels)
    fig.update_yaxes(tickmode="array", tickvals=area_labels)
    fig.write_html(output_dir / "area_topics_heatmap.html")
    fig.write_image(output_dir / "area_topics_heatmap.svg")
