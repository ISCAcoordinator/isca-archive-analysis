from typing import Callable, List, Union
from umap import UMAP
import math
import numpy as np
import pandas as pd
from warnings import warn

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff

from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from bertopic._utils import select_topic_representation
from bertopic._utils import validate_distance_matrix
from bertopic._utils import select_topic_representation

try:
    import datamapplot
    from matplotlib.figure import Figure
except ImportError:
    warn("Data map plotting is unavailable unless datamapplot is installed.")

    # Create a dummy figure type for typing
    class Figure(object):
        pass


# DEFAULT_COLORS = ["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"]
DEFAULT_COLORS = ["#D55E00", "#0072B2"]
DEFAULT_COLORS = [
    "#117733",  # Dark Green
    "#332288",  # Dark Blue
    "#DDCC77",  # Sand Yellow
    "#CC6677",  # Rose
    "#88CCEE",  # Sky Blue
    "#AA4499",  # Purple
    "#44AA99",  # Teal
    "#999933",  # Olive
    "#882255",  # Burgundy
    "#661100",  # Dark Brown
    "#6699CC",  # Muted Blue
    "#DDDDDD",  # Light Gray
    "#E17C05",  # Orange
]
colors_4 = ["#117733", "#332288", "#E17C05", "#CC6677"]  # Dark Green  # Dark Blue  # Orange  # Rose


def visualize_hierarchical_documents(
    topic_model,
    docs: List[str],
    hierarchical_topics: pd.DataFrame,
    topics: List[int] = None,
    embeddings: np.ndarray = None,
    reduced_embeddings: np.ndarray = None,
    sample: Union[float, int] = None,
    hide_annotations: bool = False,
    hide_document_hover: bool = True,
    nr_levels: int = 10,
    level_scale: str = "linear",
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Hierarchical Documents and Topics</b>",
    width: int = 1200,
    height: int = 750,
) -> go.Figure:
    """Visualize documents and their topics in 2D at different levels of hierarchy.

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                             represented by their parents and their children
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        sample: The percentage of documents in each topic that you would like to keep.
                Value can be between 0 and 1. Setting this value to, for example,
                0.1 (10% of documents in each topic) makes it easier to visualize
                millions of documents as a subset is chosen.
        hide_annotations: Hide the names of the traces on top of each cluster.
        hide_document_hover: Hide the content of the documents when hovering over
                             specific points. Helps to speed up generation of visualizations.
        nr_levels: The number of levels to be visualized in the hierarchy. First, the distances
                   in `hierarchical_topics.Distance` are split in `nr_levels` lists of distances.
                   Then, for each list of distances, the merged topics are selected that have a
                   distance less or equal to the maximum distance of the selected list of distances.
                   NOTE: To get all possible merged steps, make sure that `nr_levels` is equal to
                   the length of `hierarchical_topics`.
        level_scale: Whether to apply a linear or logarithmic (log) scale levels of the distance
                     vector. Linear scaling will perform an equal number of merges at each level
                     while logarithmic scaling will perform more mergers in earlier levels to
                     provide more resolution at higher levels (this can be used for when the number
                     of topics is large).
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
                       NOTE: Custom labels are only generated for the original
                       un-merged topics.
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:
    To visualize the topics simply run:

    ```python
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and preferred pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic and extract hierarchical topics
    topic_model = BERTopic().fit(docs, embeddings)
    hierarchical_topics = topic_model.hierarchical_topics(docs)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
    fig.write_html("path/to/file.html")
    ```

    Note:
        This visualization was inspired by the scatter plot representation of Doc2Map:
        https://github.com/louisgeisler/Doc2Map

    <iframe src="../../getting_started/visualization/hierarchical_documents.html"
    style="width:1000px; height: 770px; border: 0px;""></iframe>
    """
    topic_per_doc = topic_model.topics_

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", random_state=42).fit(
            embeddings_to_reduce
        )
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Create topic list for each level, levels are created by calculating the distance
    distances = hierarchical_topics.Distance.to_list()
    if level_scale == "log" or level_scale == "logarithmic":
        log_indices = (
            np.round(
                np.logspace(
                    start=math.log(1, 10),
                    stop=math.log(len(distances) - 1, 10),
                    num=nr_levels,
                )
            )
            .astype(int)
            .tolist()
        )
        log_indices.reverse()
        max_distances = [distances[i] for i in log_indices]
    elif level_scale == "lin" or level_scale == "linear":
        max_distances = [
            distances[indices[-1]] for indices in np.array_split(range(len(hierarchical_topics)), nr_levels)
        ][::-1]
    else:
        raise ValueError("level_scale needs to be one of 'log' or 'linear'")

    for index, max_distance in enumerate(max_distances):
        # Get topics below `max_distance`
        mapping = {topic: topic for topic in df.topic.unique()}
        selection = hierarchical_topics.loc[hierarchical_topics.Distance <= max_distance, :]
        selection.Parent_ID = selection.Parent_ID.astype(int)
        selection = selection.sort_values("Parent_ID")

        for row in selection.iterrows():
            for topic in row[1].Topics:
                mapping[topic] = row[1].Parent_ID

        # Make sure the mappings are mapped 1:1
        mappings = [True for _ in mapping]
        while any(mappings):
            for i, (key, value) in enumerate(mapping.items()):
                if value in mapping.keys() and key != value:
                    mapping[key] = mapping[value]
                else:
                    mappings[i] = False

        # Create new column
        df[f"level_{index+1}"] = df.topic.map(mapping)
        df[f"level_{index+1}"] = df[f"level_{index+1}"].astype(int)

    # Prepare topic names of original and merged topics
    trace_names = []
    topic_names = {}
    for topic in range(hierarchical_topics.Parent_ID.astype(int).max()):
        if topic < hierarchical_topics.Parent_ID.astype(int).min():
            if topic_model.get_topic(topic):
                if isinstance(custom_labels, str):
                    trace_name = f"{topic}_" + "_".join(
                        list(zip(*topic_model.topic_aspects_[custom_labels][topic]))[0][:3]
                    )
                elif topic_model.custom_labels_ is not None and custom_labels:
                    trace_name = topic_model.custom_labels_[topic + topic_model._outliers]
                else:
                    trace_name = f"{topic}_" + "_".join([word[:20] for word, _ in topic_model.get_topic(topic)][:3])
                topic_names[topic] = {
                    "trace_name": trace_name[:40],
                    "plot_text": trace_name[:40],
                }
                trace_names.append(trace_name)
        else:
            trace_name = (
                f"{topic}_"
                + hierarchical_topics.loc[hierarchical_topics.Parent_ID == str(topic), "Parent_Name"].values[0]
            )
            plot_text = "_".join([name[:20] for name in trace_name.split("_")[:3]])
            topic_names[topic] = {
                "trace_name": trace_name[:40],
                "plot_text": plot_text[:40],
            }
            trace_names.append(trace_name)

    # Prepare traces
    all_traces = []
    for level in range(len(max_distances)):
        traces = []

        # Outliers
        if topic_model._outliers:
            traces.append(
                go.Scattergl(
                    x=df.loc[(df[f"level_{level+1}"] == -1), "x"],
                    y=df.loc[df[f"level_{level+1}"] == -1, "y"],
                    mode="markers+text",
                    name="other",
                    hoverinfo="text",
                    hovertext=(
                        df.loc[(df[f"level_{level+1}"] == -1), "doc"].str.wrap(40).replace("\n", "<br>")
                        if not hide_document_hover
                        else None
                    ),
                    showlegend=False,
                    marker=dict(color="#CFD8DC", size=5, opacity=0.5),
                )
            )

        # Selected topics
        if topics:
            selection = df.loc[(df.topic.isin(topics)), :]
            unique_topics = sorted([int(topic) for topic in selection[f"level_{level+1}"].unique()])
        else:
            unique_topics = sorted([int(topic) for topic in df[f"level_{level+1}"].unique()])

        for topic in unique_topics:
            if topic != -1:
                if topics:
                    selection = df.loc[(df[f"level_{level+1}"] == topic) & (df.topic.isin(topics)), :]
                else:
                    selection = df.loc[df[f"level_{level+1}"] == topic, :]

                if not hide_annotations:
                    selection.loc[len(selection), :] = None
                    selection["text"] = ""
                    selection.loc[len(selection) - 1, "x"] = selection.x.mean()
                    selection.loc[len(selection) - 1, "y"] = selection.y.mean()
                    selection.loc[len(selection) - 1, "text"] = topic_names[int(topic)]["plot_text"]

                traces.append(
                    go.Scattergl(
                        x=selection.x,
                        y=selection.y,
                        text=selection.text if not hide_annotations else None,
                        hovertext=selection.doc if not hide_document_hover else None,
                        hoverinfo="text",
                        name=topic_names[int(topic)]["trace_name"],
                        mode="markers+text",
                        marker=dict(size=5, opacity=0.5),
                    )
                )

        all_traces.append(traces)

    # Track and count traces
    nr_traces_per_set = [len(traces) for traces in all_traces]
    trace_indices = [(0, nr_traces_per_set[0])]
    for index, nr_traces in enumerate(nr_traces_per_set[1:]):
        start = trace_indices[index][1]
        end = nr_traces + start
        trace_indices.append((start, end))

    # Visualization
    fig = go.Figure()
    for traces in all_traces:
        for trace in traces:
            fig.add_trace(trace)

    for index in range(len(fig.data)):
        if index >= nr_traces_per_set[0]:
            fig.data[index].visible = False

    # Create and add slider
    steps = []
    for index, indices in enumerate(trace_indices):
        step = dict(
            method="update",
            label=str(index),
            args=[{"visible": [False] * len(fig.data)}],
        )
        for index in range(indices[1] - indices[0]):
            step["args"][0]["visible"][index + indices[0]] = True
        steps.append(step)

    sliders = [dict(currentvalue={"prefix": "Level: "}, pad={"t": 20}, steps=steps)]

    # Add grid in a 'plus' shape
    x_range = (
        df.x.min() - abs((df.x.min()) * 0.15),
        df.x.max() + abs((df.x.max()) * 0.15),
    )
    y_range = (
        df.y.min() - abs((df.y.min()) * 0.15),
        df.y.max() + abs((df.y.max()) * 0.15),
    )
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        sliders=sliders,
        template="simple_white",
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width,
        height=height,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def visualize_distribution(
    topic_model,
    probabilities: np.ndarray,
    min_probability: float = 0.015,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Topic Probability Distribution</b>",
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Visualize the distribution of topic probabilities.

    Arguments:
        topic_model: A fitted BERTopic instance.
        probabilities: An array of probability scores
        min_probability: The minimum probability score to visualize.
                         All others are ignored.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:
    Make sure to fit the model before and only input the
    probabilities of a single document:

    ```python
    topic_model.visualize_distribution(probabilities[0])
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_distribution(probabilities[0])
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/probabilities.html"
    style="width:1000px; height: 500px; border: 0px;""></iframe>
    """
    if len(probabilities.shape) != 1:
        raise ValueError(
            "This visualization cannot be used if you have set `calculate_probabilities` to False "
            "as it uses the topic probabilities of all topics. "
        )
    if len(probabilities[probabilities > min_probability]) == 0:
        raise ValueError(
            "There are no values where `min_probability` is higher than the "
            "probabilities that were supplied. Lower `min_probability` to prevent this error."
        )

    # Get values and indices equal or exceed the minimum probability
    labels_idx = np.argwhere(probabilities >= min_probability).flatten()
    vals = probabilities[labels_idx].tolist()

    # Create labels
    if isinstance(custom_labels, str):
        labels = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in labels_idx]
        labels = ["_".join([label[0] for label in l[:4]]) for l in labels]  # noqa: E741
        labels = [label if len(label) < 30 else label[:27] + "..." for label in labels]
    elif topic_model.custom_labels_ is not None and custom_labels:
        labels = [topic_model.custom_labels_[idx + topic_model._outliers] for idx in labels_idx]
    else:
        labels = []
        for idx in labels_idx:
            words = topic_model.get_topic(idx)
            if words:
                label = [word[0] for word in words[:5]]
                label = f"<b>Topic {idx}</b>: {'_'.join(label)}"
                label = label[:40] + "..." if len(label) > 40 else label
                labels.append(label)
            else:
                vals.remove(probabilities[idx])

    # Create Figure
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            marker=dict(
                color="#C8D2D7",
                line=dict(color="#6E8484", width=1),
            ),
            orientation="h",
        )
    )

    fig.update_layout(
        xaxis_title="Probability",
        title={
            "text": f"{title}",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    )

    return fig


def visualize_document_datamap(
    topic_model,
    docs: List[str],
    topics: List[int] = None,
    embeddings: np.ndarray = None,
    reduced_embeddings: np.ndarray = None,
    custom_labels: Union[bool, str] = False,
    title: str = "Documents and Topics",
    sub_title: Union[str, None] = None,
    width: int = 1200,
    height: int = 1200,
    **datamap_kwds,
) -> Figure:
    """Visualize documents and their topics in 2D as a static plot for publication using
    DataMapPlot.

    Arguments:
        topic_model:  A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`. Documents not in these topics will be shown
                as noise points.
        embeddings:  The embeddings of all documents in `docs`.
        reduced_embeddings:  The 2D reduced embeddings of all documents in `docs`.
        custom_labels:  If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        sub_title: Sub-title of the plot.
        width: The width of the figure.
        height: The height of the figure.
        **datamap_kwds:  All further keyword args will be passed on to DataMapPlot's
                         `create_plot` function. See the DataMapPlot documentation
                         for more details.

    Returns:
        figure: A Matplotlib Figure object.

    Examples:
    To visualize the topics simply run:

    ```python
    topic_model.visualize_document_datamap(docs)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and preferred pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic
    topic_model = BERTopic().fit(docs, embeddings)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_document_datamap(docs, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)
    fig.savefig("path/to/file.png", bbox_inches="tight")
    ```
    <img src="../../getting_started/visualization/datamapplot.png",
         alt="DataMapPlot of 20-Newsgroups", width=800, height=800></img>
    """
    topic_per_doc = topic_model.topics_

    df = pd.DataFrame({"topic": np.array(topic_per_doc)})
    df["doc"] = docs
    df["topic"] = topic_per_doc

    # Extract embeddings if not already done
    if embeddings is None and reduced_embeddings is None:
        embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
    else:
        embeddings_to_reduce = embeddings

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.15, metric="cosine", random_state=42).fit(
            embeddings_to_reduce
        )
        embeddings_2d = umap_model.embedding_
    else:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc)

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = [" ".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        names = [
            f"Topic-{topic}: " + " ".join([word for word, value in topic_model.get_topic(topic)][:3])
            for topic in unique_topics
        ]

    topic_name_mapping = {topic_num: topic_name for topic_num, topic_name in zip(unique_topics, names)}
    topic_name_mapping[-1] = "Unlabelled"

    # If a set of topics is chosen, set everything else to "Unlabelled"
    if topics is not None:
        selected_topics = set(topics)
        for topic_num in topic_name_mapping:
            if topic_num not in selected_topics:
                topic_name_mapping[topic_num] = "Unlabelled"

    # Map in topic names and plot
    named_topic_per_doc = pd.Series(topic_per_doc).map(topic_name_mapping).values

    figure, axes = datamapplot.create_plot(
        embeddings_2d,
        named_topic_per_doc,
        figsize=(width / 100, height / 100),
        dpi=100,
        title=title,
        sub_title=sub_title,
        **datamap_kwds,
    )

    return figure


def visualize_topics(
    topic_model,
    topics: List[int] = None,
    top_n_topics: int = None,
    use_ctfidf: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Intertopic Distance Map</b>",
    width: int = 650,
    height: int = 650,
    colors: list[str] = DEFAULT_COLORS,
    with_slider: bool = False,
) -> go.Figure:
    """Visualize topics, their sizes, and their corresponding words.

    This visualization is highly inspired by LDAvis, a great visualization
    technique typically reserved for LDA.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        use_ctfidf: Whether to use c-TF-IDF representations instead of the embeddings from the embedding model.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:
    To visualize the topics simply run:

    ```python
    topic_model.visualize_topics()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/viz.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes_[topic] for topic in topic_list]
    if isinstance(custom_labels, str):
        words = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topic_list]
        words = ["_".join([label[0] for label in labels[:4]]) for labels in words]
        words = [label if len(label) < 30 else label[:27] + "..." for label in words]
    elif custom_labels and topic_model.custom_labels_ is not None:
        words = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topic_list]
    else:
        words = [" | ".join([word[0] for word in topic_model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])

    embeddings, c_tfidf_used = select_topic_representation(
        topic_model.c_tf_idf_,
        topic_model.topic_embeddings_,
        use_ctfidf=use_ctfidf,
        output_ndarray=True,
    )
    embeddings = embeddings[indices]

    if c_tfidf_used:
        embeddings = MinMaxScaler().fit_transform(embeddings)
        embeddings = UMAP(n_neighbors=2, n_components=2, metric="hellinger", random_state=42).fit_transform(embeddings)
    else:
        embeddings = UMAP(n_neighbors=2, n_components=2, metric="cosine", random_state=42).fit_transform(embeddings)

    # Visualize with plotly
    df = pd.DataFrame(
        {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "Topic": topic_list,
            "Words": words,
            "Size": frequencies,
        }
    )
    return _plotly_topic_visualization(df, topic_list, title, width, height, colors, with_slider=with_slider)


def _plotly_topic_visualization(
    df: pd.DataFrame,
    topic_list: List[str],
    title: str,
    width: int,
    height: int,
    colors: list[str],
    with_slider: bool = False,
):
    """Create plotly-based visualization of topics with a slider for topic selection."""

    # Prepare figure range
    x_range = (
        df.x.min() - abs((df.x.min()) * 0.15),
        df.x.max() + abs((df.x.max()) * 0.15),
    )
    y_range = (
        df.y.min() - abs((df.y.min()) * 0.15),
        df.y.max() + abs((df.y.max()) * 0.15),
    )

    # Plot topics
    # fig = go.Figure(data=go.Scattergl(
    #     x=df["x"],
    #     y=df["y"],
    #     size=df["Size"],
    #     size_max=40,
    #     template="simple_white",
    #     labels={"x": "", "y": ""},
    #     color="Topic",
    #     colorscale=colors,
    #     hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False},
    # ))
    df.Topic = df.Topic.apply(lambda x: str(int(x) + 1))
    print(df)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        size="Size",
        size_max=40,
        template="simple_white",
        # labels={"x": "", "y": ""},
        color="Topic",
        color_discrete_sequence=colors,
        hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False},
    )

    # Update hover order
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "<b>Topic %{customdata[0]}</b>",
                "%{customdata[1]}",
                "Size: %{customdata[2]}",
            ]
        )
    )

    # Stylize layout
    fig.update_layout(
        width=width,
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=20, font_family="Rockwell"),
        xaxis={"visible": False},
        yaxis={"visible": False},
        legend=dict(
            title="Topics",
            x=0.9,
            y=1,
            traceorder="reversed",
            font=dict(color="black", weight="bold"),
            bordercolor="Black",
            borderwidth=2,
        ),
    )

    # Create a slider for topic selection
    if with_slider:
        fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color="DarkSlateGrey")))

        def get_color(topic_selected):
            if topic_selected == -1:
                marker_color = ["#B0BEC5" for _ in topic_list]
            else:
                marker_color = ["red" if topic == topic_selected else "#B0BEC5" for topic in topic_list]
            return [{"marker.color": [marker_color]}]

        steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list]
        sliders = [dict(active=0, pad={"t": 50}, steps=steps)]
        fig.update_layout(
            sliders=sliders,
        )

    # Update axes ranges
    # fig['layout']['xaxis'].update(autorange = True)
    # fig['layout']['yaxis'].update(autorange = True)
    # fig.update_xaxes(range=x_range)
    # fig.update_yaxes(range=y_range)

    # Add grid in a 'plus' shape
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )

    # fig.add_annotation(x=x_range[0], y=sum() / 2, text="D1", showarrow=False, yshift=10)

    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="0.5*D2", showarrow=False, xshift=10)
    fig.data = fig.data[::-1]
    return fig


def visualize_documents(
    topic_model,
    docs: List[str],
    topics: List[int] = None,
    embeddings: np.ndarray = None,
    reduced_embeddings: np.ndarray = None,
    sample: float = None,
    hide_annotations: bool = False,
    hide_document_hover: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Documents and Topics</b>",
    width: int = 750,
    height: int = 750,
    colors: list[str] = DEFAULT_COLORS,
    use_index: bool = True,
    show_legend: bool = True,
    topic_column: str = "topic",
    topic_association_df: pd.DataFrame | None = None,
) -> Figure:
    """Visualize documents and their topics in 2D.

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        sample: The percentage of documents in each topic that you would like to keep.
                Value can be between 0 and 1. Setting this value to, for example,
                0.1 (10% of documents in each topic) makes it easier to visualize
                millions of documents as a subset is chosen.
        hide_annotations: Hide the names of the traces on top of each cluster.
        hide_document_hover: Hide the content of the documents when hovering over
                             specific points. Helps to speed up generation of visualization.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:
    To visualize the topics simply run:

    ```python
    topic_model.visualize_documents(docs)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and preferred pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic
    topic_model = BERTopic().fit(docs, embeddings)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_documents(docs, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
    fig.write_html("path/to/file.html")
    ```

    <iframe src="../../getting_started/visualization/documents.html"
    style="width:1000px; height: 800px; border: 0px;""></iframe>
    """
    topic_per_doc = topic_model.topics_

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]

    if topic_association_df is not None:
        topic_association_df["topic"] = topic_association_df["topic"] - 1  # FIXME: this is hardcoded for now
        df = df.merge(topic_association_df[["topic", topic_column]], on="topic")

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=6, min_dist=0.0, metric="cosine", random_state=42).fit(
            embeddings_to_reduce
        )
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_topics = set(df[topic_column])
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    # elif topic_model.custom_labels_ is not None and custom_labels:
    #     names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        # names = [
        #     f"{topic}_" + "_".join([word for word, value in topic_model.get_topic(topic)][:3])
        #     for topic in unique_topics
        # ]
        names = df[topic_column]

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics)  # .difference(topics)
    # if len(non_selected_topics) == 0:
    #     non_selected_topics = [-1]

    selection = df.loc[df[topic_column].isin(non_selected_topics), ["doc", topic_column, "x", "y"]]
    selection["text"] = ""
    selection.loc[len(selection), :] = [
        None,
        None,
        selection.x.mean(),
        selection.y.mean(),
        "Other documents",
    ]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode="markers+text",
            name="other",
            showlegend=False,
            marker=dict(color="#CFD8DC", size=5, opacity=0.5),
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df[topic_column] == topic, ["doc", topic_column, "x", "y"]]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [
                    None,
                    None,
                    selection.x.mean(),
                    selection.y.mean(),
                    name,
                ]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode="markers+text",
                    name=topic if use_index else name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5, color=colors[topic % len(colors)]),
                )
            )

    # Add grid in a 'plus' shape
    x_range = (
        df.x.min() - abs((df.x.min()) * 0.15),
        df.x.max() + abs((df.x.max()) * 0.15),
    )
    y_range = (
        df.y.min() - abs((df.y.min()) * 0.15),
        df.y.max() + abs((df.y.max()) * 0.15),
    )
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        width=width,
        height=height,
    )

    if not show_legend:
        fig.update_layout(showlegend=False)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def visualize_barchart(
    topic_model,
    topics: List[int] = None,
    top_n_topics: int = 8,
    n_words: int = 5,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Topic Word Scores</b>",
    width: int = 250,
    height: int = 250,
    autoscale: bool = False,
    colors: list[str] = DEFAULT_COLORS,
) -> go.Figure:
    """Visualize a barchart of selected topics.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.
        autoscale: Whether to automatically calculate the height of the figures to fit the whole bar text

    Returns:
        fig: A plotly figure

    Examples:
    To visualize the barchart of selected topics
    simply run:

    ```python
    topic_model.visualize_barchart()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>
    """

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    # Initialize figure
    if isinstance(custom_labels, str):
        subplot_titles = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        subplot_titles = ["_".join([label[0] for label in labels[:4]]) for labels in subplot_titles]
        subplot_titles = [label if len(label) < 30 else label[:27] + "..." for label in subplot_titles]
    elif topic_model.custom_labels_ is not None and custom_labels:
        subplot_titles = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_xaxes=False,
        horizontal_spacing=0.1,
        vertical_spacing=0.4 / rows if rows > 1 else 0,
        subplot_titles=subplot_titles,
    )

    # Add barchart for each topic
    row = 1
    column = 1
    for i_topic, topic in enumerate(topics):
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores, y=words, orientation="h", marker_color=colors[i_topic % len(colors)]),
            row=row,
            col=column,
        )

        if autoscale:
            if len(words) > 12:
                height = 250 + (len(words) - 12) * 11

            if len(words) > 9:
                fig.update_yaxes(tickfont=dict(size=(height - 140) // len(words)))

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def visualize_hierarchy(
    topic_model,
    orientation: str = "left",
    topics: List[int] = None,
    top_n_topics: int = None,
    use_ctfidf: bool = True,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Hierarchical Clustering</b>",
    width: int = 1000,
    height: int = 600,
    hierarchical_topics: pd.DataFrame = None,
    linkage_function: Callable[[csr_matrix], np.ndarray] = None,
    distance_function: Callable[[csr_matrix], csr_matrix] = None,
    color_threshold: int = 1,
) -> go.Figure:
    """Visualize a hierarchical structure of the topics.

    A ward linkage function is used to perform the
    hierarchical clustering based on the cosine distance
    matrix between topic embeddings (either c-TF-IDF or the embeddings from the embedding model).

    Arguments:
        topic_model: A fitted BERTopic instance.
        orientation: The orientation of the figure.
                     Either 'left' or 'bottom'
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        use_ctfidf: Whether to calculate distances between topics based on c-TF-IDF embeddings. If False, the embeddings
                    from the embedding model are used.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
                       NOTE: Custom labels are only generated for the original
                       un-merged topics.
        title: Title of the plot.
        width: The width of the figure. Only works if orientation is set to 'left'
        height: The height of the figure. Only works if orientation is set to 'bottom'
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                             represented by their parents and their children.
                             NOTE: The hierarchical topic names are only visualized
                             if both `topics` and `top_n_topics` are not set.
        linkage_function: The linkage function to use. Default is:
                          `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
                          NOTE: Make sure to use the same `linkage_function` as used
                          in `topic_model.hierarchical_topics`.
        distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                           `lambda x: 1 - cosine_similarity(x)`.
                            You can pass any function that returns either a square matrix of
                            shape (n_samples, n_samples) with zeros on the diagonal and
                            non-negative values or condensed distance matrix of shape
                            (n_samples * (n_samples - 1) / 2,) containing the upper
                            triangular of the distance matrix.
                           NOTE: Make sure to use the same `distance_function` as used
                           in `topic_model.hierarchical_topics`.
        color_threshold: Value at which the separation of clusters will be made which
                         will result in different colors for different clusters.
                         A higher value will typically lead in less colored clusters.

    Returns:
        fig: A plotly figure

    Examples:
    To visualize the hierarchical structure of
    topics simply run:

    ```python
    topic_model.visualize_hierarchy()
    ```

    If you also want the labels visualized of hierarchical topics,
    run the following:

    ```python
    # Extract hierarchical topics and their representations
    hierarchical_topics = topic_model.hierarchical_topics(docs)

    # Visualize these representations
    topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    ```

    If you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_hierarchy()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/hierarchy.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    if distance_function is None:
        distance_function = lambda x: 1 - cosine_similarity(x)

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, "ward", optimal_ordering=True)

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Select embeddings
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])

    # Select topic embeddings
    embeddings = select_topic_representation(topic_model.c_tf_idf_, topic_model.topic_embeddings_, use_ctfidf)[0][
        indices
    ]

    # Annotations
    if hierarchical_topics is not None and len(topics) == len(freq_df.Topic.to_list()):
        annotations = _get_annotations(
            topic_model=topic_model,
            hierarchical_topics=hierarchical_topics,
            embeddings=embeddings,
            distance_function=distance_function,
            linkage_function=linkage_function,
            orientation=orientation,
            custom_labels=custom_labels,
        )
    else:
        annotations = None

    # wrap distance function to validate input and return a condensed distance matrix
    distance_function_viz = lambda x: validate_distance_matrix(distance_function(x), embeddings.shape[0])
    # Create dendogram
    fig = ff.create_dendrogram(
        embeddings,
        orientation=orientation,
        distfun=distance_function_viz,
        linkagefun=linkage_function,
        hovertext=annotations,
        color_threshold=color_threshold,
    )

    # Create nicer labels
    axis = "yaxis" if orientation == "left" else "xaxis"
    if isinstance(custom_labels, str):
        new_labels = [
            [[str(x), None]] + topic_model.topic_aspects_[custom_labels][x] for x in fig.layout[axis]["ticktext"]
        ]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]
    elif topic_model.custom_labels_ is not None and custom_labels:
        new_labels = [
            topic_model.custom_labels_[topics[int(x)] + topic_model._outliers] for x in fig.layout[axis]["ticktext"]
        ]
    else:
        new_labels = [
            [[str(topics[int(x)]), None]] + topic_model.get_topic(topics[int(x)]) for x in fig.layout[axis]["ticktext"]
        ]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

    # Stylize layout
    fig.update_layout(
        plot_bgcolor="#ECEFF1",
        template="plotly_white",
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
    )

    # Stylize orientation
    if orientation == "left":
        fig.update_layout(
            height=200 + (15 * len(topics)),
            width=width,
            yaxis=dict(tickmode="array", ticktext=new_labels),
        )

        # Fix empty space on the bottom of the graph
        y_max = max([trace["y"].max() + 5 for trace in fig["data"]])
        y_min = min([trace["y"].min() - 5 for trace in fig["data"]])
        fig.update_layout(yaxis=dict(range=[y_min, y_max]))

    else:
        fig.update_layout(
            width=200 + (15 * len(topics)),
            height=height,
            xaxis=dict(tickmode="array", ticktext=new_labels),
        )

    if hierarchical_topics is not None:
        for index in [0, 3]:
            axis = "x" if orientation == "left" else "y"
            xs = [data["x"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            ys = [data["y"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            hovertext = [data["text"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    marker_color="black",
                    hovertext=hovertext,
                    hoverinfo="text",
                    mode="markers",
                    showlegend=False,
                )
            )
    return fig


def _get_annotations(
    topic_model,
    hierarchical_topics: pd.DataFrame,
    embeddings: csr_matrix,
    linkage_function: Callable[[csr_matrix], np.ndarray],
    distance_function: Callable[[csr_matrix], csr_matrix],
    orientation: str,
    custom_labels: bool = False,
) -> List[List[str]]:
    """Get annotations by replicating linkage function calculation in scipy.

    Arguments:
        topic_model: A fitted BERTopic instance.
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                             represented by their parents and their children.
                             NOTE: The hierarchical topic names are only visualized
                             if both `topics` and `top_n_topics` are not set.
        embeddings: The c-TF-IDF matrix on which to model the hierarchy
        linkage_function: The linkage function to use. Default is:
                          `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
                          NOTE: Make sure to use the same `linkage_function` as used
                          in `topic_model.hierarchical_topics`.
        distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                           `lambda x: 1 - cosine_similarity(x)`.
                            You can pass any function that returns either a square matrix of
                            shape (n_samples, n_samples) with zeros on the diagonal and
                            non-negative values or condensed distance matrix of shape
                            (n_samples * (n_samples - 1) / 2,) containing the upper
                            triangular of the distance matrix.
                           NOTE: Make sure to use the same `distance_function` as used
                           in `topic_model.hierarchical_topics`.
        orientation: The orientation of the figure.
                     Either 'left' or 'bottom'
        custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       NOTE: Custom labels are only generated for the original
                       un-merged topics.

    Returns:
        text_annotations: Annotations to be used within Plotly's `ff.create_dendogram`
    """
    df = hierarchical_topics.loc[hierarchical_topics.Parent_Name != "Top", :]

    # Calculate distance
    X = distance_function(embeddings)
    X = validate_distance_matrix(X, embeddings.shape[0])

    # Calculate linkage and generate dendrogram
    Z = linkage_function(X)
    P = sch.dendrogram(Z, orientation=orientation, no_plot=True)

    # store topic no.(leaves) corresponding to the x-ticks in dendrogram
    x_ticks = np.arange(5, len(P["leaves"]) * 10 + 5, 10)
    x_topic = dict(zip(P["leaves"], x_ticks))

    topic_vals = dict()
    for key, val in x_topic.items():
        topic_vals[val] = [key]

    parent_topic = dict(zip(df.Parent_ID, df.Topics))

    # loop through every trace (scatter plot) in dendrogram
    text_annotations = []
    for index, trace in enumerate(P["icoord"]):
        fst_topic = topic_vals[trace[0]]
        scnd_topic = topic_vals[trace[2]]

        if len(fst_topic) == 1:
            if isinstance(custom_labels, str):
                fst_name = f"{fst_topic[0]}_" + "_".join(
                    list(zip(*topic_model.topic_aspects_[custom_labels][fst_topic[0]]))[0][:3]
                )
            elif topic_model.custom_labels_ is not None and custom_labels:
                fst_name = topic_model.custom_labels_[fst_topic[0] + topic_model._outliers]
            else:
                fst_name = "_".join([word for word, _ in topic_model.get_topic(fst_topic[0])][:5])
        else:
            for key, value in parent_topic.items():
                if set(value) == set(fst_topic):
                    fst_name = df.loc[df.Parent_ID == key, "Parent_Name"].values[0]

        if len(scnd_topic) == 1:
            if isinstance(custom_labels, str):
                scnd_name = f"{scnd_topic[0]}_" + "_".join(
                    list(zip(*topic_model.topic_aspects_[custom_labels][scnd_topic[0]]))[0][:3]
                )
            elif topic_model.custom_labels_ is not None and custom_labels:
                scnd_name = topic_model.custom_labels_[scnd_topic[0] + topic_model._outliers]
            else:
                scnd_name = "_".join([word for word, _ in topic_model.get_topic(scnd_topic[0])][:5])
        else:
            for key, value in parent_topic.items():
                if set(value) == set(scnd_topic):
                    scnd_name = df.loc[df.Parent_ID == key, "Parent_Name"].values[0]

        text_annotations.append([fst_name, "", "", scnd_name])

        center = (trace[0] + trace[2]) / 2
        topic_vals[center] = fst_topic + scnd_topic

    return text_annotations


def visualize_topics_over_time(
    topic_model,
    topics_over_time: pd.DataFrame,
    top_n_topics: int = None,
    topics: List[int] = None,
    normalize_frequency: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Topics over Time</b>",
    width: int = 600,
    height: int = 450,
    colors: list[str] = DEFAULT_COLORS,
    use_index: bool = True,
    use_scatter: bool = True,
) -> go.Figure:
    """Visualize topics over time.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_over_time: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Returns:
        A plotly.graph_objects.Figure including all traces

    Examples:
    To visualize the topics over time, simply run:

    ```python
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    topic_model.visualize_topics_over_time(topics_over_time)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/trump.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if isinstance(custom_labels, str):
        topic_names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        topic_names = ["_".join([label[0] for label in labels[:4]]) for labels in topic_names]
        topic_names = [label if len(label) < 30 else label[:27] + "..." for label in topic_names]
        topic_names = {key: topic_names[index] for index, key in enumerate(topic_model.topic_labels_.keys())}
    elif topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {
            key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()
        }
    else:
        topic_names = {
            key: value[:40] + "..." if len(value) > 40 else value for key, value in topic_model.topic_labels_.items()
        }
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(
            go.Scatter(
                x=trace_data.Timestamp,
                y=y,
                mode="lines" if not use_scatter else "markers+lines",
                marker_color=colors[index % len(colors)],
                hoverinfo="text",
                name=index if use_index else topic_name,
                hovertext=[f"<b>Topic {topic}</b><br>Words: {word}" for word in words],
            )
        )

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        legend=dict(
            title="Topics",
        ),
    )
    return fig
