# ISCA Archive analysis toolkit

The goal of this repository is to propose a set of tools to conduct the analysis of the metadata stored in the [ISCA Archive](https://www.isca-archive.org).

## How to install

### Install the package

This repository is not yet in a published form, so you need to use the couple pip/git URL to install it

```sh
pip install git+https://www.github.com/ISCACoordinator/isca-archive-analysis.git
```

If you want to contribute, please fork the repository and install the tool using

```sh
pip install -e .[dev]
```

### Install the resources

It is also necessary to download some preliminary resources

```sh
    python -m spacy download en_core_web_sm
```

### LLM support

If you want to use LLAMA, it is also necessary to install the following packages:

```sh
    pip install huggingface_hub transformers
```

You will need an API token (info pasted from https://colab.research.google.com/drive/1QCERSMUjqGetGGujdrvv_6_EeoIcd_9M?usp=sharing#scrollTo=OwK3QLOTaQE9):

 1. Create a HuggingFace account: https://huggingface.co/
 2. Apply for Llama 2 access: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
 3. Get your HuggingFace token: https://huggingface.co/settings/tokens

## How to run

## Topic Analysis

In order to run the topic analysis, the following ISCA Archive directory structure is assumed:

```
- published_archive
    |
    +---- archive  # Contains the actual online part of the archive
    +---- metadata # Contains the JSON metadata which is used to "regenerate" the HTML part if necessary
```

### Analysis helper

We provide an helper to run the topic analysis `helpers/topic_analysis.sh` which runs a default topic analysis on Interspeech papers.
To run the analysis, simply call the following command

```sh
    bash helpers/topic_analysis.sh
```

Each step can activated/deactivated in the command line.
For example the following command disable the dataset generation (i.e., assumes it has already been done), and forces the wordcloud and the topic analysis to be ran

```sh
    GEN_DATASET=FALSE WORD_CLOUD=TRUE DEFAULT_TOPIC=TRUE bash -x helpers/topic_analysis.sh
```

For more information, please refer to the helper file
