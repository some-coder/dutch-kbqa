"""
Methods to integrate new translations to the Python-interpreted version of the JSON QA dataset.
"""


import csv
import re
import requests

from constants import ORIGINAL_DATA_FILE, TRANSLATIONS_FILE
from convert import Language, EntityLocatingTechnique, NaturalLanguageQuestion, SPARQLAnswer, QAPair, qa_pairs_from_json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BRACKETS_PATTERN: str = '\\[[^]]+]'
MOD_PATTERN: str = '(M[0-9]+)'


def _english_entities_version(qa_pair: QAPair, language: Language) -> str:
	"""
	Internal use. Obtains a with-bracket question string in a non-English language, with English entities.

	This method currently only supports Dutch. We naively assume that the order of bracketed entities is always the
	same for the English and non-English language, but this may of course not be true. That's currently a potential
	bug, but Dutch may be less susceptible to this than, say, Hebrew (with its right-to-left orientation).

	:param qa_pair: The QA pair to work on.
	:param language: A non-English language whose entities to replace with English ones.
	:returns: The English entities representation.
	:raises ValueError: When `language` is set English.
	"""
	supported_languages: Tuple[Language, ...] = (Language.DUTCH,)
	if language not in supported_languages:
		raise ValueError('[_english_entities_version] Language \'%s\' not supported!' % (language.value,))
	english = qa_pair.q.form(Language.ENGLISH, EntityLocatingTechnique.WITH_BRACKETS)
	non_english = qa_pair.q.form(language, EntityLocatingTechnique.WITH_BRACKETS)
	return re.sub(BRACKETS_PATTERN, '[%s]', non_english) % tuple(t[1:-1] for t in re.findall(BRACKETS_PATTERN, english))


def _mod_pattern_entities_version(qa_pair: QAPair, language: Language) -> str:
	"""
	Internal use. Obtains a mod pattern entities question string in a non-English language.

	:param qa_pair: The QA pair to work on.
	:param language: A non-English language to create a mod pattern entities question representation of.
	:returns: The mod pattern entities representation.
	:raises ValueError: When the QA pair does not have an English language version with a `WITH_BRACKETS` version, or
		when the requested language does not have a `WITH_BRACKETS` version of the question.
	"""
	mpe = qa_pair.q.form(Language.ENGLISH, EntityLocatingTechnique.MOD_PATTERN_ENTITIES)
	return \
		re.sub(
			BRACKETS_PATTERN,
			'%s',
			qa_pair.q.form(language, EntityLocatingTechnique.WITH_BRACKETS)) % \
		tuple(re.findall(MOD_PATTERN, mpe))


def freebase_mid_to_wikidata_and_natural_languages(
		freebase_mid: str,
		languages: Tuple[Language, ...]) -> Tuple[str, Dict[Language, str]]:
	"""
	Obtains the Wikidata entity ID plus natural language name(s) for the given Freebase MID.

	We regard the first response as being the ground truth. This, of course, need not necessarily be appropriate.

	:param freebase_mid: The Freebase MID to use.
	:param languages: The language(s) to get the entity labels for.
	:returns: A two-tuple of (1) a Wikidata entity ID, and (2) a mapping from languages to labels.
	"""
	if len(languages) == 0:
		raise ValueError('[freebase_mid_to_wikidata_natural_languages] At least one language needs to be given!')
	url: str = 'https://query.wikidata.org/sparql'
	query: str = \
		'%s {\n  %s .\n  %s .\n  %s .\n}' % \
		(
			'SELECT * WHERE',
			'?subject wdt:P646 "/m/%s"' % freebase_mid.replace('m.', ''),
			'?subject rdfs:label ?name',
			'filter(lang(?name) = "%s")'
		)
	wd_id: Optional[str] = None
	names: Dict[Language, str] = {}
	for language in languages:
		req = requests.get(url, params={'format': 'json', 'query': query % language.value}).json()
		best_binding: Dict[str, Dict[str, str]] = req['results']['bindings'][0]
		wd_id = re.sub('(http)(s)?(://www.wikidata.org/entity/)', '', best_binding['subject']['value'])
		names[language] = best_binding['name']['value']
	return wd_id, names


def integrate(qa_pairs: List[QAPair], location: Path, language: Language, english_entities: bool = False) -> None:
	"""
	Integrates the new translations in the CSV file at the specified location to the supplied list of QA pairs.

	This method does not check whether `language` is already in use in `qa_pairs`, nor does it check for duplicates in
	the CSV file. Be wary of this! The CSV should be of the format: 'QA pairs index', 'CFQ ID', 'translation'. No
	header row should be present, just data rows. Further, the `EntityLocatingTechnique` that is assumed to be used
	for each `translation` is `WITH_BRACKETS`.

	:param qa_pairs: The QA pairs to add new translations to.
	:param location: The location of the CSV file from which to grab new translations.
	:param language: The language used in the entries of the CSV file.
	:param english_entities: Whether to use English entities for integrating this non-English language. Default: false.
	"""
	with open(location, 'r') as handle:
		reader = csv.reader(handle)
		error_count: int = 0
		for row in reader:
			row: Tuple[int, int, str] = (int(row[0]), int(row[1]), row[2])  # index, CFQ ID, translation
			if qa_pairs[row[0]].identifier != row[1]:
				raise RuntimeError('[integrate] Inconsistent index and CFQ ID: %d and %d!' % (row[0], row[1]))
			qa_pairs[row[0]].q.representations[language] = {}
			qa_pairs[row[0]].q.representations[language][EntityLocatingTechnique.WITH_BRACKETS] = \
				_english_entities_version(qa_pairs[row[0]], language) if english_entities else row[2]
			try:
				qa_pairs[row[0]].q.representations[language][EntityLocatingTechnique.MOD_PATTERN_ENTITIES] = \
					_mod_pattern_entities_version(qa_pairs[row[0]], language)
			except TypeError:
				error_count += 1
				print(
					'(%4d) %s' %
					(row[0], qa_pairs[row[0]].q.form(Language.ENGLISH, EntityLocatingTechnique.WITH_BRACKETS)))
		print('\n\nERROR COUNT: %3d.' % (error_count,))


if __name__ == '__main__':
	print(freebase_mid_to_wikidata_and_natural_languages('m.0f8l9c', (Language.ENGLISH, Language.DUTCH)))
