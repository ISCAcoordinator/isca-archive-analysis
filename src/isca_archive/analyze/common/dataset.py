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
		self._root_dir: pathlib.Path = root_dir
		self._list_conferences: list[str] | None = list_conference
		self._list_files: list[pathlib.Path] = []

		# Ensure the list of conference contains something
		if self._list_conferences is None:
			self._list_conferences = []
			for path in self._root_dir.iterdir():
				if path.is_dir():
					if path.name not in ["pdfs", "resources"]:
						self._list_conferences.append(path.name)

		self._generate_file_list()

	def __len__(self) -> int:
		"""Get the number of items from the archive

		Returns
		-------
		int
			the number of items in the archive

		"""
		return len(self._list_files)

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
		html_file = self._root_dir / conference / html_basename
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
			conf_dir = self._root_dir / conf_name
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
			pdf_path = self._root_dir / conference / href
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
