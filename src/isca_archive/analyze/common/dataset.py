# General python
import re
from typing import Any
import pathlib
import logging

# Torch helpers
from torch.utils.data import Dataset

# Data processing
from lxml import html, etree
import fitz
import bibtexparser
import json
import pandas as pd
from pkg_resources import resource_filename


class ISCAArchiveProcessorDataset(Dataset):
	"""Helper class to load the raw ISCA Archive and process it

	This class should mainly be used to generate a TSV file which can
	then be loaded by ISCAArchiveProcessedDataset.
	"""

	def __init__(self, root_dir: pathlib.Path, list_conference: list[str] | None = None):
		"""Initialisation

		Parameters
		----------
		root_dir : pathlib.Path
			the root directory of the archive
		list_conference : list[str] | None
			the list of conferences to focus on (if None, all conferences are loaded!)
		"""
		self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)
		self._metadata_dir: pathlib.Path = root_dir/"metadata"
		self._html_root_dir: pathlib.Path = root_dir/"archive"
		self._list_conferences: list[str] | None = list_conference
		self._list_entries: list[tuple[dict]] = list()
		self._fill_entries()

	def _fill_entries(self):
		if self._list_conferences is None:
			raise Exception("oupsy")

		for conf in self._list_conferences:
			conf_metadata_file = self._metadata_dir/f"{conf}.json"
			with open(conf_metadata_file, 'r') as f_conf:
				conf_metadata = json.load(f_conf)
				conf_info = dict()
				conf_info["title"] = conf_metadata["title"]
				conf_info["series"] = conf_metadata["series"]
				conf_info["year"] = conf_metadata["year"]
				conf_info["id"] = conf
				for cur_paper_id, cur_paper_entry in conf_metadata["papers"].items():
					cur_paper_entry["paper_id"] = cur_paper_id
					self._list_entries.append((conf_info, cur_paper_entry))

	def __len__(self) -> int:
		"""Get the number of items from the archive

		Returns
		-------
		int
			the number of items in the archive

		"""
		return len(self._list_entries)

	def __getitem__(self, idx: int) -> dict[str, Any]:
		"""Get one item from the archive

		Parameters
		----------
		idx : int
			the index of wanted item

		Returns
		-------
		dict[str, Any]
			The dictionnary containing the values for the wanted item

		Raises
		------
		Exception
			If any loading issues happen
		"""
		(conf_info, paper_entry) = self._list_entries[idx]

		conf_id = conf_info["id"]
		paper_id = paper_entry["paper_id"]
		title = paper_entry["title"]
		abstract = "\n".join(paper_entry["abstract"])

		pdf_file = None
		try:
			pdf_file = self._get_pdf(conf_id, paper_id)
		except Exception as ex:
			self._logger.warning(f'The submission ID "{paper_id}" doesn\'t seem valid (pdf extraction failure): {ex}')

		references = []
		try:
			if pdf_file is not None:
				references = self._extract_biblio_from_article(pdf_file)
		except Exception as ex:
			self._logger.warning(
				f'The submission "{paper_id}" doesn\'t seem valid (references extraction failure): {ex}'
			)

		is_area = None
		try:
			is_area = self._get_area(conf_id, paper_entry)
			self._logger.info(f"==> The area for {paper_id} is {is_area}")
		except Exception as ex:
			self._logger.warning(
				f'The submission "{paper_id}" doesn\'t any area assigned: {ex}'
			)

		return {
			"title": title,
			"abstract": abstract,
			"serie": conf_info["series"],
			"year": conf_info["year"],
			"pdf_path": pdf_file,
			"references": references,
			"is_area": is_area
		}

	def _get_area(self, conference: str, paper: dict[str, Any]) -> str | None:

		# Prepare the paper id
		paper_id = paper["original"]
		if not paper_id.startswith("i"):
			paper_id = f"{int(paper_id):04d}"

		# NOTE: could be optimized to not load this everytime !
		conf_resource = pathlib.Path(resource_filename("isca_archive", f'resources/is_areas/{conference}.tsv'))
		if not conf_resource.is_file():
			raise Exception("The area file doesn't exist")

		df_conf_areas= pd.read_csv(
			conf_resource,
			sep="\t",
			dtype={'paper_id': str}
		)

		area_id = df_conf_areas.loc[df_conf_areas.paper_id == paper_id, "area_id"]
		if len(area_id) == 0:
			raise Exception(f"The paper doesn't have any area assigned to it ({paper['original']})")
		else:
			area_id = area_id.item()

		if area_id == "00":
			return None

		df_area_labels = pd.read_csv(
			resource_filename("isca_archive", 'resources/is_area_labels.tsv'),
			sep="\t"
		)


		label = df_area_labels.loc[(df_area_labels.primary_id == area_id) & pd.isna(df_area_labels.secondary_id), "label"].item()
		return label



	def _get_pdf(self, conference: str, paper_id: str) -> pathlib.Path:
		"""Internal helper to get the PDF file associated to the item

		Parameters
		----------
		conference : str
			the conference name of the item
		paper_id : str
		    the paper identifier (not the submission number, the ISCA archive one)

		Returns
		-------
		pathlib.Path
			the path to the PDF File

		Raises
		------
		FileNotFoundError
		    if the PDF file does not exist
		Exception
			if the PDF is corrupted

		"""
		pdf_path = self._html_root_dir / conference / f"{paper_id}.pdf"
		if pdf_path.exists():
			doc = fitz.open(pdf_path)
			if doc.page_count <= 0:
				raise Exception("the PDF doesn't contain any pages!")
			return pdf_path
		else:
			raise FileNotFoundError(f"The following PDF file does not exist: {pdf_path}")

	def _find_references(self, doc: fitz.fitz.Document, start_page: int, num_headers: bool) -> list[str]:
		"""Internal helper to determine where the reference section starts and load all the lines of this section in a list

		The start of the reference section is determined if the line only contains the term "Reference" prefixed (or not) by the number of the section

		Parameters
		----------
		doc: fitz.fitz.Document
		   The loaded PDF document
		start_page: int
		   The starting page for the parsing (counting is decremental!)
		num_headers: bool
		   Flag to impose/ignore the section number

		Returns
		-------
		list[str]
		   The list of lines contained in the section. If empty, it is likely that the section hasn't been found!
		"""
		id_page = start_page - 1
		last_page = doc.load_page(id_page)
		lines = last_page.get_text().split("\n")
		id_ref_section = -1

		# Define which format to use to determine the section header
		if num_headers:
			# p_ref_section = re.compile("^[0-9]+[^a-zA-Z]* References[ ]*$")
			p_ref_section = "^[0-9]+[^a-zA-Z]* References[ ]*$"
		else:
			p_ref_section = "^[ ]*References[ ]*$"

		# Search for the header (and lines' list generation)
		while (id_ref_section < 0) and (id_page > 0):
			for i_l, l in enumerate(lines):
				if not l.strip():
					continue
				m = re.match(p_ref_section, l, flags=re.IGNORECASE)
				if m:
					id_ref_section = i_l

			# If no section is found, try the previous page
			if id_ref_section < 0:
				id_page -= 1
				cur_page = doc.load_page(id_page)
				cur_lines = cur_page.get_text().split("\n")
				lines = cur_lines + lines

		if id_page != 0:
			return lines[id_ref_section + 1 :]
		else:
			return []

	def _retrieve_references(self, lines_ref_section: list[str]) -> tuple[list[str], list[str]]:
		"""Internal helper to extract the references from the lines extracted from the reference section

		A reference is assumed to start with an ID (format = [<number]).
		This function can also extract only the titles which are assumed to be surrounded by the delimiters "“" and "”".

		Parameters
		----------
		lines_ref_section: list[str]
			The lines of the reference section

		Returns
		-------
		list[str]
			the list of references or the list of titles depending on the value of the parameters `only_titles`
		"""
		list_refs = []
		cur_ref = ""
		for _, l in enumerate(lines_ref_section):
			if not l.strip():
				continue

			if l.startswith("["):
				cur_ref = re.sub(r"^\[[0-9]+\][ ]*", "", cur_ref)
				list_refs.append(cur_ref)
				cur_ref = ""

			cur_ref += re.sub("-$", "", l)

		# Deal with last references
		cur_ref = re.sub(r'^\[[0-9]+\][ ]*', "", cur_ref)
		list_refs.append(cur_ref)

		# Extract titles
		ref_titles = []
		p_title = re.compile(".*“([^”]*)”.*")
		for ref in list_refs:
			# “Low delay noise reduction and dereverberation for hearing aid,”
			m_title = p_title.match(ref)
			if m_title:
				ref_titles.append(m_title.group(1))

		return ref_titles, list_refs

	def _extract_biblio_from_article(self, paper_path: pathlib.Path) -> list[str]:
		"""Internal helper to extract the bibliography from a given paper.

		Parameters
		----------
		paper_path: pathlib.Path
			The path to the PDF of the paper

		Returns
		-------
		list[str]
			The list of the titles of the references used by the given paper

		Raises
		------
		Exception
			if the reference section can't be found

		"""
		doc = fitz.open(paper_path)
		n_pages = doc.page_count

		lines = self._find_references(doc, n_pages, num_headers=True)

		if not lines:
			lines = self._find_references(doc, n_pages, num_headers=False)

		if not lines:
			raise Exception("No reference page")

		ref_titles, _ = self._retrieve_references(lines)

		return ref_titles

class ExternalISCAArchiveProcessorDataset(ISCAArchiveProcessorDataset):
	"""Helper class to load the raw ISCA Archive and process it

	This class should mainly be used to generate a TSV file which can
	then be loaded by ISCAArchiveProcessedDataset.
	"""

	def __init__(self, root_dir: pathlib.Path, list_conference: list[str] | None = None):
		super().__init__(root_dir, list_conference)
		self._archive_dir = root_dir/"archive"

	def __getitem__(self, idx: int) -> dict[str, Any]:
		"""Get one item from the archive

		Parameters
		----------
		idx : int
			the index of wanted item

		Returns
		-------
		dict[str, Any]
			The dictionnary containing the values for the wanted item

		Raises
		------
		Exception
			If any loading issues happen
		"""
		(conference, html_basename) = self._list_files[idx]

		# Load the HTML file
		html_file = self._archive_dir / conference / html_basename
		html_tree = html.parse(html_file)
		html_root = html_tree.getroot()

		# Get information
		try:
			title = self._get_title(html_root)
		except Exception as ex:
			raise Exception(f'File "{html_basename}" doesn\'t seem valid (title processing failure): {ex}')

		abstract = self._get_abstract(html_root)

		pdf_file = None
		try:
			pdf_file = self._get_pdf(html_root, conference)
		except Exception as ex:
			if False:
				raise Exception(f'File "{html_basename}" doesn\'t seem valid (pdf extraction failure): {ex}')
			else:
				self._logger.warning(f'File "{html_basename}" doesn\'t seem valid (pdf extraction failure): {ex}')

		references = []
		try:
			if pdf_file is not None:
				references = self._extract_biblio_from_article(pdf_file)
		except Exception as ex:
			if False:
				raise Exception(f'File "{html_basename}" doesn\'t seem valid (references extraction failure): {ex}')
			else:
				self._logger.warning(
					f'File "{html_basename}" doesn\'t seem valid (references extraction failure): {ex}'
				)

		serie, year = conference.split("_")
		return {
			"title": title,
			"abstract": abstract,
			"serie": serie,
			"year": year,
			"html_file_name": html_file,
			"pdf_path": pdf_file,
			"references": references,
		}

	def _generate_file_list(self) -> list[tuple[str, str]]:
		"""Internal helper to generate the list of items (1 item = 1 file)

		Returns
		-------
		list[tuple[str, str]]
			The list of files in form of a tuple (the conference name, the name of the item/file)
		"""
		self._list_files = []
		for conf_name in self._list_conferences:
			conf_dir = self._archive_dir / conf_name
			for path in conf_dir.iterdir():
				if path.name.endswith(".html") and (path.name != "index.html"):
					self._list_files.append((conf_name, path.name))

	def _get_abstract(self, html_root: etree.Element) -> str:
		"""Internal helper to extract the abstract of the item

		Parameters
		----------
		html_root : etree.Element
			the HTML element of the current item

		Returns
		-------
		str
			the abstract

		Raises
		------
		Exception
			If there was any issue during the parsing of the HTML file
			=> no abstract!

		"""
		# Define the XPath expression to find a div with a specific class
		abstract_xpath = ".//div[@id='abstract']/p"

		# Find all matching div elements
		candidates = html_root.findall(abstract_xpath)

		if candidates:
			try:
				text = candidates[0].text.strip().replace("\n", " ")
			except Exception as ex:
				# There is some formatting, so just pick the text here!
				text = candidates[0].xpath("string()").strip().replace("\n", " ")
			return text
		else:
			raise Exception("No abstract has been found")

	def _get_pdf(self, html_root: etree.Element, conference: str) -> pathlib.Path:
		"""Internal helper to get the PDF file associated to the item

		Parameters
		----------
		html_root : etree.Element
			the HTML element corresponding to the item
		conference : str
			the conference name of the item

		Returns
		-------
		pathlib.Path
			the path to the PDF File

		Raises
		------
		Exception
			if no PDF were found or the PDF is corrupted

		"""
		# Define the XPath expression to find a div with a specific class
		pdf_xpath = ".//div[@id='content']/div/a/@href"

		# Find all matching div elements
		href = html_root.xpath(pdf_xpath)[0]

		if href:
			pdf_path = self._archive_dir / conference / href
			doc = fitz.open(pdf_path)
			if doc.page_count <= 0:
				raise Exception("the PDF doesn't contain any pages!")
			return pdf_path
		else:
			raise Exception("No PDF link has been found")

	def _get_title(self, html_root: etree.Element) -> str:
		"""Internal helper to extract the title of the item

		Parameters
		----------
		html_root : etree.Element
			the HTML element corresponding to the item

		Returns
		-------
		str
			the title of the item

		Raises
		------
		Exception
			Parsing issue => couldn't find a title!
		"""
		# Define the XPath expression to find a div with a specific class
		title_xpath = ".//div[@id='global-info']/h3"

		# Find all matching div elements
		candidates = html_root.findall(title_xpath)

		if candidates:
			try:
				text = candidates[0].text.strip().replace("\n", " ")
			except Exception as ex:
				# There is some formatting, so just pick the text here!
				text = candidates[0].xpath("string()").strip().replace("\n", " ")
			return text
		else:
			raise Exception("No title has been found")


class ISCAArchiveProcessedDataset(Dataset):
	"""Dataset wrapper for the ISCA Archive dataframe"""

	def __init__(
		self,
		input: pathlib.Path | pd.DataFrame,
		series: list[str] | None = None,
		years: list[int] | None = None,
	):
		"""Initialisation

		Just set the dataframe

		Parameters
		----------
		input : Union[pathlib.Path|pd.DataFrame]
			either the file from which to load the dataframe or the already loaded dataframe itself
		"""
		if isinstance(input, pd.DataFrame):
			self.df = input
		else:
			self.df = pd.read_csv(input, sep="\t")

		# Filter series if required
		if series is not None:
			self.df = self.df[self.df.serie.str.lower().isin(series)]

		# Filter years if required
		if years is not None:
			self.df = self.df[self.df.year.isin(years)]

		# Ensure the index works out
		self.df.reset_index(inplace=True, drop=True)

	def __len__(self) -> int:
		"""Get the number of articles in the ISCA archive

		Returns
		-------
		int
			The number of available articles

		"""
		return len(self.df.index)

	def __getitem__(self, idx: int) -> dict[str, Any]:
		"""Return a given item in a form of a dictionary

		Parameters
		----------
		idx : int
			the index of the item

		Returns
		-------
		dict[str, Any]
			the dictionnary containing the value of the wanted item.
		"""
		return self.df.iloc[idx].to_dict()
