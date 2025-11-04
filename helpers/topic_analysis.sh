#!/bin/bash


##################################################################################
## Define control variables
##################################################################################

# Process control flags
: ${GEN_DATASET:=TRUE}
: ${WORD_CLOUD:=FALSE}
: ${DEFAULT_TOPIC:=TRUE}
: ${ZERO_SHOT_PRIMARY:=TRUE}
: ${ZERO_SHOT_SECONDARY:=FALSE}

# Directory/files helpers
: ${ARCHIVE_DIR:=published/}
: ${INPUT_ANALYSIS_DIR:=input_analysis}
: ${ROOT_OUTPUT:=topic_analysis_output}
: ${CONFIG_FILE:=config.ini}

# Additional helpers
: ${CLUSTERING_ALGORITHM:="balanced_kmeans"}
: ${IGNORE_ISCA_STOPWORDS:=TRUE}
: ${IGNORE_DATA_KEYWORDS:=FALSE}
: ${IGNORE_ML_KEYWORDS:=FALSE}
: ${USE_LLAMA:=FALSE}
: ${USE_FULL_TEXT:=FALSE}
: ${USE_ISCA_KEYWORDS_ONLY:=TRUE}
: ${USE_TITLE:=TRUE}
: ${VERBOSE_LEVEL:=DEBUG}
: ${SERIES:="interspeech"} # NOTE: for multiple series: "interspeech,eurospeech,icslp"
: ${YEARS:="2013-2024"}
: ${NB_TOPICS:=13} # NOTE: default value based on Interspeech 2021 -> 2024
: ${ONLY_VISUALISE:=FALSE}
: ${TOPIC_CLUSTER:=""}
: ${SEED:=42}

##################################################################################
## Prepare arguments and define constants
##################################################################################

# Define some constants
LLAMA_TOKEN=$(grep "llama_api_token" $CONFIG_FILE | sed 's/.*=//g')

# Define arguments based on options (NOTE series and years are imposed)
EXTRACT_ARGS=()
LOG_ARGS=()
VISUALIZE_ARGS=()
ARGS=(-s ${SERIES})
ARGS+=(-y ${YEARS})
TOPICS_ARGS=(--nb-topics ${NB_TOPICS} --clustering-algorithm $CLUSTERING_ALGORITHM --seed "${SEED}")
[ "$IGNORE_ISCA_STOPWORDS" = "TRUE" ]                 && ARGS+=(--ignore-isca-stopwords)
[ "$IGNORE_ML_KEYWORDS" = "TRUE" ]                    && ARGS+=(--ignore-ml-keywords)
[ "$IGNORE_DATA_KEYWORDS" = "TRUE" ]                  && ARGS+=(--ignore-data-keywords)
[ "$USE_LLAMA" = "TRUE" ] && [ "$LLAMA_TOKEN" != "" ] && TOPICS_ARGS+=(--use-llama "$LLAMA_TOKEN")
[ "$USE_FULL_TEXT" = "TRUE" ]                         && EXTRACT_ARGS+=(--full-text)
[ "$USE_ISCA_KEYWORDS_ONLY" = "TRUE" ]                && EXTRACT_ARGS+=(--isca-keywords)
[ "$USE_TITLE" = "TRUE" ]                             && EXTRACT_ARGS+=(--use-title)
[ "$VERBOSE_LEVEL" = "INFO" ]                         && LOG_ARGS=(-v)
[ "$VERBOSE_LEVEL" = "DEBUG" ]                        && LOG_ARGS=(-vv)

# FIXME: this creates some issues for what ever reason
# [ "${TOPIC_CLUSTER}" != "" ]                          && VISUALIZE_ARGS+=(-i "${TOPIC_CLUSTER}")

##################################################################################
## actual process
##################################################################################

# Generate dataframe
mkdir -p $ROOT_OUTPUT/logs
if [[ "${GEN_DATASET}" == "TRUE" ]]; then
    archive_analysis $LOG_ARGS -l $ROOT_OUTPUT/logs/dataset.log generate_dataset ${EXTRACT_ARGS[@]} $ARCHIVE_DIR $INPUT_ANALYSIS_DIR/lists/all_conf.txt $ROOT_OUTPUT/isca_df.json
fi

# Simple Word Cloud
if [[ "${WORD_CLOUD}" == "TRUE" ]]; then
    time archive_analysis $LOG_ARGS -l $ROOT_OUTPUT/logs/word_cloud.log generate_word_cloud ${ARGS[@]} $ROOT_OUTPUT/isca_df.json $ROOT_OUTPUT/word_cloud
fi

# Overall
if [[ "${DEFAULT_TOPIC}" == "TRUE" ]]; then
    if [[ "${ONLY_VISUALISE}" != "TRUE" ]]; then
        time archive_analysis $LOG_ARGS -l $ROOT_OUTPUT/logs/topic_analysis.log analyze_topics ${ARGS[@]} ${TOPICS_ARGS[@]} $ROOT_OUTPUT/isca_df.json $ROOT_OUTPUT/unconstrained
    fi

    archive_analysis visualize_topics ${VISUALIZE_ARGS[@]} $ROOT_OUTPUT/unconstrained/docs_with_topics.json $ROOT_OUTPUT/unconstrained/model $ROOT_OUTPUT/unconstrained/figures
fi

# Zero-Shot (Primary Area - FIXME - not yet working)
if [[ "${ZERO_SHOT_PRIMARY}" == "TRUE" ]]; then
    if [[ "${ONLY_VISUALISE}" != "TRUE" ]]; then
        time archive_analysis $LOG_ARGS -l $ROOT_OUTPUT/logs/primary_area.log zero_shot_topic ${ARGS[@]} $ROOT_OUTPUT/isca_df.json $ROOT_OUTPUT/imposed/primary
    fi

    archive_analysis visualize_topics $ROOT_OUTPUT/imposed/primary/docs_with_topics.json $ROOT_OUTPUT/imposed/primary/model $ROOT_OUTPUT/imposed/primary/figures
fi

# Zero-Shot (Secondary Area - FIXME - not yet working)
if [[ "${ZERO_SHOT_SECONDARY}" == "TRUE" ]]; then
    if [[ "${ONLY_VISUALISE}" != "TRUE" ]]; then
        time archive_analysis $LOG_ARGS -l $ROOT_OUTPUT/logs/secondary_area.log zero_shot_topic ${ARGS[@]} -S $ROOT_OUTPUT/isca_df.json $ROOT_OUTPUT/imposed/secondary
    fi

    archive_analysis visualize_topics $ROOT_OUTPUT/imposed/secondary/docs_with_topics.json $ROOT_OUTPUT/imposed/secondary/model $ROOT_OUTPUT/imposed/secondary/figures
fi
