"""
Methods to integrate new translations to the Python-interpreted version of the JSON QA dataset.
"""


import csv
import re
import requests
import time

from constants import ORIGINAL_DATA_FILE, ORIGINAL_TRANSLATIONS_FILE, MODIFIED_TRANSLATIONS_FILE
from convert import Language, EntityLocatingTechnique, QAPair, qa_pairs_from_json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# querying
TOO_MANY_REQUESTS_CODE: int = 429
WIKIDATA_QUERY_WAIT: float = float(60 // 30)
WIKIDATA_RETRY_WAIT: float = 1e0

# regular expressions
BRACKETS_PATTERN: str = '\\[[^]]+]'
MOD_PATTERN: str = '(M[0-9]+)'
FREEBASE_MID_PATTERN: str = '( m\\.)[a-zA-Z0-9_]{2,7}'


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
	print('Searching for labels for Freebase MID %s...' % (freebase_mid,))
	for language in languages:
		time.sleep(WIKIDATA_QUERY_WAIT)
		req = requests.get(url, params={'format': 'json', 'query': query % language.value})
		while req.status_code == TOO_MANY_REQUESTS_CODE:
			time.sleep(WIKIDATA_RETRY_WAIT)
			print('[freebase_mid_to_wikidata_and_natural_languages] Need to retry. Hold on for a second...')
			req = requests.get(url, params={'format': 'json', 'query': query % language.value})
		req = req.json()
		best_binding: Dict[str, Dict[str, str]] = req['results']['bindings'][0]
		wd_id = re.sub('(http)(s)?(://www.wikidata.org/entity/)', '', best_binding['subject']['value'])
		names[language] = best_binding['name']['value']
	print('\tFound! In English: %s.' % (names[Language.ENGLISH],))
	return wd_id, names


def _freebase_machine_ids(qa_pairs: List[QAPair], languages: Tuple[Language, ...]) -> Dict[str, Dict[Language, str]]:
	"""
	Internal use. Collects the set of unidentified Freebase MIDs from the QA pairs, and makes a map for it.

	:param qa_pairs: The QA pairs to search for Freebase MIDs in.
	:param languages: The natural language label languages to obtain.
	:returns: A mapping from Freebase MIDs to language representations of the MIDs.
	"""
	machine_ids: Tuple[str, ...] = tuple()
	for index, qa_pair in enumerate(qa_pairs):
		rep = qa_pair.q.form(Language.ENGLISH, EntityLocatingTechnique.WITH_BRACKETS)
		for match in re.finditer(FREEBASE_MID_PATTERN, rep):
			if match.group()[1:] not in machine_ids:
				machine_ids += match.group()[1:],
	m: Dict[str, Dict[Language, str]] = {}
	for mid in machine_ids:
		m[mid] = freebase_mid_to_wikidata_and_natural_languages(mid, languages)[1]
	return m


def _unidentified_freebase_entities(qa_pair: QAPair) -> Set[str]:
	"""
	Internal use. Collects all Freebase MIDs in the question that haven't been converted yet. May be empty.

	:param qa_pair: The QA pair to search within.
	:returns: The set. May be empty.
	"""
	s: Set[str] = set()
	for language in qa_pair.q.representations.keys():
		rep = qa_pair.q.form(language, EntityLocatingTechnique.WITH_BRACKETS)
		for match in re.finditer(FREEBASE_MID_PATTERN, rep):
			s.add(match.group()[1:])  # convert to the RDF version of the MIDs
	return s


def _resolve_unidentified_freebase_entities(qa_pair: QAPair, mid_map: Dict[str, Dict[Language, str]]) -> None:
	"""
	Internal use. Replaces any Freebase MIDs in the QA pair with proper entity names.

	:param qa_pair: The QA pair to work on.
	:param mid_map: The mapping from Freebase MIDs and languages to representations in said languages.
	"""
	for language in qa_pair.q.representations.keys():
		rep = qa_pair.q.form(language, EntityLocatingTechnique.WITH_BRACKETS)
		qa_pair.q.representations[language][EntityLocatingTechnique.WITH_BRACKETS] = \
			re.sub(FREEBASE_MID_PATTERN, lambda match: ' [%s]' % (mid_map[match.group()[1:]][language],), rep)


def _resolve_unidentified_freebase_entities_csv(
		location: Path,
		mid_map: Dict[str, Dict[Language, str]],
		language: Language) -> List[Tuple[int, int, str]]:
	"""
	Internal use. Replaces any Freebase MIDs in the rows of the CSV file with proper entity names.

	:param location: The location of the CSV file to alter.
	:param mid_map: A mapping from Freebase MIDs and languages to natural language labels.
	:param language: The language of the entries in the CSV file.
	:returns: A list of rows. Each row stores a(n) (1) index, (2) CFQ ID, and (3) a natural language expression.
	"""
	resolved: List[Tuple[int, int, str]] = []
	with open(location, 'r') as handle:
		reader = csv.reader(handle)
		for line in reader:
			index: int = int(line[0])
			cfq_id: int = int(line[1])
			exp: str = \
				re.sub(FREEBASE_MID_PATTERN, lambda match: ' [%s]' % (mid_map[match.group()[1:]][language]), line[2])
			resolved.append((index, cfq_id, exp))
	return resolved


def _replaced_html_character_references(exp: str) -> str:
	"""
	Internal use. Returns the expression with the HTML character references ('&#hhhh;') replaced.

	:param exp: The expression to work on.
	:returns: The fixed expression.
	"""
	match = re.search('(&)((quot|amp|apos|lt|gt)|((#)[0-9]{2,4}))(;)?', exp)
	while match is not None:
		sp: Tuple[int, int] = match.span()
		mt = match.group() + (';' if match.group()[-1] != ';' else '')
		if mt == '&quot;':
			exp = exp[:sp[0]] + '"' + exp[sp[1]:]
		elif mt == '&amp;':
			exp = exp[:sp[0]] + '&' + exp[sp[1]:]
		elif mt == '&apos;':
			exp = exp[:sp[0]] + '\'' + exp[sp[1]:]
		elif mt == 'lt':
			exp = exp[:sp[0]] + '<' + exp[sp[1]:]
		elif mt == 'gt':
			exp = exp[:sp[0]] + '>' + exp[sp[1]:]
		else:
			# must be a non-XML predefined HTML character code
			exp = exp[:sp[0]] + chr(int(match.group()[2:-1])) + exp[sp[1]:]
		match = re.search('(&)((quot|amp|apos|lt|gt)|((#)[0-9]{2,4}))(;)?', exp)
	return exp


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
		for r in reader:
			r: Tuple[int, int, str] = (int(r[0]), int(r[1]), r[2])  # index, CFQ ID, translation
			if qa_pairs[r[0]].identifier != r[1]:
				raise RuntimeError('[integrate] Inconsistent index and CFQ ID: %d and %d!' % (r[0], r[1]))
			qa_pairs[r[0]].q.representations[language] = {}
			qa_pairs[r[0]].q.representations[language][EntityLocatingTechnique.WITH_BRACKETS] = \
				_english_entities_version(qa_pairs[r[0]], language) if english_entities else r[2]
			# _resolve_unidentified_freebase_entities(qa_pairs[row[0]])
			qa_pairs[r[0]].q.representations[language][EntityLocatingTechnique.MOD_PATTERN_ENTITIES] = \
				_mod_pattern_entities_version(qa_pairs[r[0]], language)


def preprocessed_questions_answers(location: Path, mm: Dict[str, Dict[Language, str]]) -> List[QAPair]:
	"""
	Obtains the QA pairs at the supplied location, properly pre-processed.

	:param location: The location to get the QA pairs from.
	:param mm: A mapping from Freebase MIDs and languages to labels.
	:returns: The pre-processed QA pairs.
	"""
	# get the original data
	questions_answers = qa_pairs_from_json(location)
	# repair the questions in the QA pairs object
	for question_answer in questions_answers:
		_resolve_unidentified_freebase_entities(question_answer, mm)
		for lang in question_answer.q.representations.keys():
			question_answer.q.representations[lang][EntityLocatingTechnique.WITH_BRACKETS] = \
				_replaced_html_character_references(
					question_answer.q.representations[lang][EntityLocatingTechnique.WITH_BRACKETS]
				)
			question_answer.q.representations[lang][EntityLocatingTechnique.MOD_PATTERN_ENTITIES] = \
				_replaced_html_character_references(
					question_answer.q.representations[lang][EntityLocatingTechnique.MOD_PATTERN_ENTITIES]
				)
	return questions_answers


def preprocessed_csv_file(
		location: Path,
		mm: Dict[str, Dict[Language, str]],
		target_lang: Language,
		target_location: Path) -> None:
	"""
	Preprocesses the CSV file for integration.

	:param location: The location of the CSV file.
	:param mm: A mapping from Freebase MIDs and languages to labels.
	:param target_lang: The language of expressions in the CSV file.
	:param target_location: The location to save the modified CSV file to. (We won't overwrite the original.)
	"""
	rows = _resolve_unidentified_freebase_entities_csv(location, mm, target_lang)
	if target_lang != Language.DUTCH:
		raise NotImplementedError('[preprocessed_csv_file] Language \'%s\' not (yet) supported!' % (target_lang,))
	with open(target_location, 'w') as hdl:
		writer = csv.writer(hdl)
		for index, row in enumerate(rows):
			if index == 5374:
				# Strange Google Cloud Translate behaviour: skip a symbol in one of the Freebase MIDs.
				row = (row[0], row[1], row[2].replace('m. 07sc', '[Verenigd Koninkrijk]'))
			writer.writerow((row[0], row[1], _replaced_html_character_references(row[2])))


def qa_pairs_with_dutch(
		qa_loc: Path,
		original_csv_loc: Path,
		modified_csv_loc: Path) -> List[QAPair]:
	"""
	Yields the original JSON data as a Python-interpreted object, including the Dutch language for questions.

	:param qa_loc: The location of the original QAs in JSON.
	:param original_csv_loc: The location of the original Dutch translations CSV file.
	:param modified_csv_loc: The location of where to save the modified Dutch translations CSV file.
	:returns: The modified QA pairs.
	"""
	freebase_mm = _freebase_machine_ids(qa_pairs_from_json(qa_loc), tuple(lang for lang in Language))
	qa_pairs = preprocessed_questions_answers(qa_loc, freebase_mm)
	preprocessed_csv_file(
		original_csv_loc,
		freebase_mm,
		Language.DUTCH,
		modified_csv_loc)
	integrate(qa_pairs, modified_csv_loc, Language.DUTCH, english_entities=False)
	return qa_pairs


if __name__ == '__main__':
	with_dutch = qa_pairs_with_dutch(
		Path(ORIGINAL_DATA_FILE),
		Path(ORIGINAL_TRANSLATIONS_FILE),
		Path(MODIFIED_TRANSLATIONS_FILE))
	print(with_dutch[5374].q.form(Language.DUTCH, EntityLocatingTechnique.WITH_BRACKETS))
