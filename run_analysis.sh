#!/bin/bash -xe

SEEDS=(42 128 256 512 1024 1337)
SEEDS+=(16384 65536 524288)
SEEDS=(512)
# CLUSTERING_ALGORITHM="kmeans"
CLUSTERING_ALGORITHM="balanced_kmeans"
START_YEAR="2015"

for SEED in "${SEEDS[@]}"
do
    RUN_ID="${CLUSTERING_ALGORITHM}_${START_YEAR}_${SEED}"
    ROOT_OUTPUT="runs/run_${RUN_ID}/abstract"
    TOPIC_CLUSTER_TSV="${ROOT_OUTPUT}/unconstrained/model/topic2label.tsv"
    mkdir -p "${ROOT_OUTPUT}/unconstrained/figures"

    WORD_CLOUD=FALSE TOPIC_CLUSTER="$TOPIC_CLUSTER_TSV" SEED="$SEED" YEARS="${START_YEAR}-2025" CLUSTERING_ALGORITHM="$CLUSTERING_ALGORITHM" ONLY_VISUALISE=FALSE USE_ISCA_KEYWORDS_ONLY=FALSE ROOT_OUTPUT=$ROOT_OUTPUT bash -x helpers/topic_analysis.sh

    # python to_sort_scripts/extract_topic_theme.py "${ROOT_OUTPUT}/unconstrained/model/topics.json" "$TOPIC_CLUSTER_TSV"
    # # python to_sort_scripts/compute_direct_citation_graph.py "output_file" "${ROOT_OUTPUT}/unconstrained/docs_with_topics.json" "$TOPIC_CLUSTER_TSV" "${ROOT_OUTPUT}/unconstrained/figures/citation_graph.pdf"
    # python to_sort_scripts/plot_sankey.py "${ROOT_OUTPUT}/unconstrained/docs_with_topics.json" "$TOPIC_CLUSTER_TSV" "${ROOT_OUTPUT}/unconstrained/figures"
    # python to_sort_scripts/plot_topics_over_time.py "${ROOT_OUTPUT}/unconstrained/docs_with_topics.json" "$TOPIC_CLUSTER_TSV" "${ROOT_OUTPUT}/unconstrained/figures/topics_over_time_by_topics.pdf"
    # python to_sort_scripts/plot_themes_over_time.py "${ROOT_OUTPUT}/unconstrained/docs_with_topics.json" "$TOPIC_CLUSTER_TSV" "${ROOT_OUTPUT}/unconstrained/figures/topics_over_time_by_theme.pdf"
done


########################################################################


# TOPIC_CLUSTER=$PWD/topic_clusters_indexed.tsv ONLY_VISUALISE=FALSE USE_ISCA_KEYWORDS_ONLY=TRUE ROOT_OUTPUT="topic_analysis_output/indexed_terms" bash -x helpers/topic_analysis.sh
# # isca_archive_analyze topic_area topic_analysis_output/indexed_terms/unconstrained/docs_with_topics.json topic_analysis_output/indexed_terms/unconstrained/figures
# (
#     cd topic_analysis_output/indexed_terms/unconstrained/figures
#     ls -1 *.pdf | xargs -I {} -P 10 pdfcrop {}
#     rename 's/-crop.pdf/.pdf/g' *
# )

# TOPIC_CLUSTER=$PWD/topic_clusters_indexed_title.tsv ONLY_VISUALISE=TRUE USE_ISCA_KEYWORDS_ONLY=TRUE USE_TITLE=TRUE ROOT_OUTPUT="topic_analysis_output/indexed_terms_title" bash -x helpers/topic_analysis.sh
# # isca_archive_analyze topic_area topic_analysis_output/indexed_terms/unconstrained/docs_with_topics.json topic_analysis_output/indexed_terms/unconstrained/figures
# (
#     cd topic_analysis_output/indexed_terms_title/unconstrained/figures
#     ls -1 *.pdf | xargs -I {} -P 10 pdfcrop {}
#     rename 's/-crop.pdf/.pdf/g' *
# )
