from typing import Set
import nltk

nltk.download("stopwords")

research_stopwords = {
	"abstract",
	"article",
	"author",
	"authors",
	"background",
	"conclusion",
	"data",
	"discussion",
	"experiment",
	"figure",
	"findings",
	"hypothesis",
	"introduction",
	"keywords",
	"method",
	"methods",
	"objective",
	"objectives",
	"paper",
	"papers",
	"problem",
	"publication",
	"published",
	"purpose",
	"question",
	"references",
	"related",
	"result",
	"results",
	"studies",
	"study",
	"table",
	"tables",
	"theory",
	"topic",
	"use"
	"used",
	"uses",
	"using",
}


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
	"network"
	"predict",
	"prediction",
	"predictive",
	"regression",
	"supervised",
	"training",
	"unlabeled",
	"unsupervised",
}


def generate_stop_words(
	with_research: bool = False, with_data_analysis: bool = False, with_ml: bool = False
) -> Set[str]:
	stopwords = set(nltk.corpus.stopwords.words("english"))

	if with_research:
		stopwords.update(research_stopwords)

	if with_data_analysis:
		stopwords.update(data_analysis_stopwords)

	if with_ml:
		stopwords.update(machine_learning_stopwords)

	return stopwords
