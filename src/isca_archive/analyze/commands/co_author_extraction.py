import pathlib
import argparse

import networkx as nx
import numpy as np
import plotly.express as px

# File / Dataset
from isca_archive.analyze.common.dataset import ISCAArchiveProcessedDataset
from isca_archive.analyze.common.args import parse_range
from isca_archive.analyze.common.author import gimme_max, generate_collaboration_dict, generate_collaboration_graph

def add_subparsers(subparsers):
    parser = subparsers.add_parser(
        "generate_co_author_graph", help="Generate the co-author graph"
    )

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


    # Add arguments
    parser.add_argument("input_dataframe", help="the ISCA Archive processed dataframe file")
    parser.add_argument("output_directory", help="The output directory")
    parser.set_defaults(func=main)


def main(args: argparse.Namespace):
    # Load the dataset
    series = args.serie_subset
    if series is not None:
        series = series.split(",")

    years = args.year_subset

    dataset = ISCAArchiveProcessedDataset(args.input_dataframe, series=series, years=years)

    # Generate helper dictionnaries
    dict_author_decomp = dict()
    dict_author = dict()
    dict_paper = dict()
    dict_bert_topic = dict()
    dict_isca_topic = dict()
    for paper_info in dataset:
        paper_id = paper_info["paper_id"]
        bert_topic_id = paper_info["BERTopic"]
        isca_topic_id = paper_info["author_isca_topic"]
        if isca_topic_id == "Show and Tell":
            isca_topic_id = ""
        elif isca_topic_id != "":
            isca_topic_id = int(isca_topic_id.split(".")[0])
        dict_paper[paper_id] = set()

        if bert_topic_id not in dict_bert_topic:
            dict_bert_topic[bert_topic_id] = set()

        if (isca_topic_id != "") and (isca_topic_id not in dict_isca_topic):
            dict_isca_topic[isca_topic_id] = set()

        for author in paper_info["authors"]:
            author_str = " ".join(author)
            if not author_str in dict_author:
                dict_author[author_str] = set()
            dict_author[author_str].add(paper_id)
            dict_paper[paper_id].add(author_str)
            dict_bert_topic[bert_topic_id].add(author_str)
            if isca_topic_id != "":
                dict_isca_topic[isca_topic_id].add(author_str)
            dict_author_decomp[author_str] = author



    # Some preliminary information about the authors
    print(f"The dataset contains {len(dict_author.keys())} distinct authors")
    print(f"The average ratio author/paper is {len(dict_author.keys())/len(dict_paper.keys())}")
    big_author = gimme_max(dict_author).pop()
    print(big_author)
    print(f"The most prolific author is {big_author} with {len(dict_author[big_author])}: {dict_author[big_author]}")

    # Generate area matrix
    nb_topics = np.max(list(dict_isca_topic.keys()))
    collab_isca_areas = np.zeros((nb_topics, nb_topics))
    for i_area in range(1, nb_topics+1):
        for j_area in range(0, i_area-1):
            collab_isca_areas[i_area-1, j_area] = len(dict_isca_topic[i_area].intersection(dict_isca_topic[j_area+1])) / len(dict_isca_topic[i_area])
        #     print(f"{collab_isca_areas[j_area, i_area-1]:02.1f}", end="\t")
        # print("")

    nb_topics = np.max(list(dict_bert_topic.keys()))
    collab_bert_areas = np.zeros((nb_topics, nb_topics))
    for i_area in range(1, nb_topics+1):
        for j_area in range(0, i_area-1):
            collab_bert_areas[i_area-1, j_area] = len(dict_bert_topic[i_area].intersection(dict_bert_topic[j_area+1])) / len(dict_bert_topic[i_area])
        #     print(f"{collab_bert_areas[j_area, i_area-1]:02.1f}", end="\t")
        # print("")


    # Move to collaboration
    collaboration_dict = generate_collaboration_dict(dict_paper)
    big_collaborator = gimme_max(collaboration_dict)
    big_collaborator = big_collaborator.pop()
    print(f"The most prolific collaboration is {big_collaborator} with {len(collaboration_dict[big_collaborator])}")

    # NOTE: ensure everything is saved
    # Save everything
    output_dir = pathlib.Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot heatmap
    fig = px.imshow(
        collab_isca_areas,
        labels=dict(x="ISCA Area", y="ISCA Area", color="Ratio"),
        text_auto=True,  # Automatically display values in cells
    )

    fig.update_layout(title="Number of collaborations normalised by number of publications (ISCA Area)")
    # fig.update_layout(
    #     title="Heatmap of Counts",
    #     xaxis_title="BERTopic",
    #     yaxis_title="Author ISCA Topic",
    # )

    fig.write_html(output_dir/"collaboration_area_heatmap.html")
    fig.write_image(output_dir/"collaboration_area_heatmap.svg")

    # Plot heatmap
    fig = px.imshow(
        collab_bert_areas,
        labels=dict(x="BERTopic", y="BERTopic", color="Ratio"),
        text_auto=True,  # Automatically display values in cells
    )

    fig.update_layout(title="Number of collaborations normalised by number of publications (BERTopic)")
    # fig.update_layout(
    #     title="Heatmap of Counts",
    #     xaxis_title="BERTopic",
    #     yaxis_title="Author ISCA Topic",
    # )

    fig.write_html(output_dir/"collaboration_bert_heatmap.html")
    fig.write_image(output_dir/"collaboration_bert_heatmap.svg")


    # # # Generate JSON graph
    # # graph = generate_collaboration_graph_reaction(collaboration_dict)
    # # with open("graph.json", "w") as f_out:
    # #     json.dump(graph, f_out, indent=2)

    # G = generate_collaboration_graph(collaboration_dict)

    # # Find connected components
    # connected_components = list(nx.connected_components(G))

    # # Print the number of clusters
    # num_clusters = len(connected_components)
    # print("Number of clusters:", num_clusters)

    # # Sort the connected components by size
    # sorted_components = sorted(connected_components, key=len, reverse=True)

    # # Print information about the largest connected component
    # for i in range(len(sorted_components)):
    #     # print("\nNodes in the largest connected component:", largest_component)
    #     current_cluster = sorted_components[i]
    #     print(f"Size of the cluster {i:04d}: {len(current_cluster)}")
