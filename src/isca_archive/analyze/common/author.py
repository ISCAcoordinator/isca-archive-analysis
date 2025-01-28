import itertools
import networkx as nx


_END = "_end_"

def gimme_max(the_dict: dict[str, list[str]]) -> set[str]:
	max_key = set()
	max_length = -1

	for key, value in the_dict.items():
		if len(value) > max_length:
			max_length = len(value)
			max_key = set([key])
		elif len(value) == max_length:
			max_key.add(key)
	return max_key

def generate_collaboration_dict(the_dict: dict[str, list[str]]) -> dict[str, set[str]]:
	collaboration_dict: dict[str, set[str]] = dict()

	for key, value in the_dict.items():
		if len(value) == 1:
			continue
		if "" in value:
			continue
		combinations = sorted(set(itertools.combinations(value, 2)))
		for comb in combinations:
			comb = "_".join(comb)
			if comb not in collaboration_dict:
				collaboration_dict[comb] = set()
			collaboration_dict[comb].add(key)
	return collaboration_dict

def generate_collaboration_graph_reaction(collaboration_dict: dict[str, set[str]]) -> dict:
	collaboration_graph = {"nodes": set(), "links": list()}
	for current_collaboration, papers in collaboration_dict.items():

		authors = current_collaboration.split("_")
		for author in authors:
			if author not in collaboration_graph["nodes"]:
				collaboration_graph["nodes"].add(author)

		collaboration_graph["links"].append({
			"source": authors[0],
			"target": authors[1],
			"value": len(papers)
		})

	# Fix the authors to be compatible with the react format
	collaboration_graph["nodes"] = [{"id": author, "group": 1} for author in collaboration_graph["nodes"]]
	return collaboration_graph

def generate_collaboration_graph(collaboration_dict: dict[str, set[str]]) -> nx.Graph:
	graph = nx.Graph()

	for current_collaboration, papers in collaboration_dict.items():
		authors = current_collaboration.split("_")
		for author in authors:
			graph.add_node(author)

		graph.add_edge(authors[0], authors[1])

	return graph

def make_trie(authors):
	root = dict()
	for author in authors.keys():
		current_dict = root
		for letter in author:
			current_dict = current_dict.setdefault(letter, {})
		current_dict[_END] = authors[author]
	return root

def get_author(author_trie, author):
	output = author_trie
	for l in author:
		output = output[l]
	if _END in output:
		return output[_END]
	return output
