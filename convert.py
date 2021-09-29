"""
Methods for converting a JSON file to a Python-interpretable list of question-answer pairs.
"""


import json
import re

from constants import ORIGINAL_DATA_FILE
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union


JSONQAPair = Dict[str, Union[int, str, bool]]


class Language(Enum):
	"""
	A language. Language codes follow the ISO-639-1 standard.
	"""
	ENGLISH = 'en'
	HEBREW = 'he'
	KANNADA = 'kn'
	CHINESE = 'zh'  # this is simplified Chinese, as opposed to traditional Chinese
	DUTCH = 'nl'


def language_is_left_to_right(language: Language) -> bool:
	"""
	Determines whether the requested language is left-to-right. If not, it's right-to-left.

	:param language: The language to determine the writing direction for.
	:returns: The question's answer.
	"""
	return language != Language.HEBREW


class EntityLocatingTechnique(Enum):
	"""
	A technique for locating and (possibly) masking entities within an expression (a question or answer).
	"""
	NONE = 'none'
	PATTERN = 'pattern'
	MOD_PATTERN_ENTITIES = 'mod-pattern-entities'
	WITH_BRACKETS = 'with-brackets'


class NaturalLanguageQuestion:
	"""
	A question in one or more natural languages, and with one or more entity locating techniques applied to it.
	"""

	def __init__(self, representations: Dict[Language, Dict[EntityLocatingTechnique, str]]) -> None:
		"""
		Constructs a question represented in multiple language representations.

		:param representations: A mapping from language-locating technique pairs to expressions.
		"""
		self.representations = deepcopy(representations)
		self._pre_process_representations()

	def _pre_process_representations(self) -> None:
		"""
		Pre-processes the representations at construction of the question. Internal use: don't call directly.
		"""
		# remove question marks
		for language, techniques_map in self.representations.items():
			for technique, expr in techniques_map.items():
				reg_exp: str = '[?\\uFF1F]+'
				self.representations[language][technique] = re.sub(
					reg_exp + '$' if language_is_left_to_right(language) else '^' + reg_exp, '', expr)

	def form(self, language: Language, technique: EntityLocatingTechnique, question_mark: bool = True) -> str:
		"""
		Retrieves the representation of the question in the given language and locating technique.

		:param language: The natural language to represent the question in.
		:param technique: The technique with which to locate named entities.
		:param question_mark: Whether to include a question mark. Defaults to `True`.
		:raises ValueError: When the combination of language and locating technique is not present.
		:returns: The requested form of the natural language question.
		"""
		question: str
		try:
			if technique == EntityLocatingTechnique.NONE:
				question = re.sub(
					pattern='[\\[\\]]',
					repl='',
					string=self.representations[language][EntityLocatingTechnique.WITH_BRACKETS])
			else:
				question = self.representations[language][technique]
		except KeyError:
			raise ValueError(
				'[%s] Could not find the language-technique combo \'%s\'-\'%s\'!' %
				(self.__class__.__name__, language.value, technique.value))
		if question_mark:
			# Take into account writing direction. Chinese is assumed to be simplified and left-to-right.
			return question + '?' if language_is_left_to_right(language) else '?' + question
		else:
			return question

	def __str__(self) -> str:
		"""
		Returns a string representation of the natural language question.

		:returns: The string representation.
		"""
		return 'NLQuestion(...)'

	def as_json_map(self) -> Dict[str, Any]:
		"""
		Represents the question as a JSON-ready mapping.

		:returns: A mapping that is ready to be converted to JSON.
		"""
		return \
			{
				lan.value: {tec.value: self.form(lan, tec, question_mark=True) for tec in self.representations[lan]}
				for lan in self.representations.keys()
			}


class SPARQLAnswer:
	"""
	An answer represented as a SPARQL query to the answer. Has one or more entity linking techniques applied to it.
	"""

	def __init__(self, representations: Dict[EntityLocatingTechnique, str]) -> None:
		"""
		Constructs an answer represented in SPARQL.

		:param representations: A mapping from entity locating techniques to SPARQL representations.
		"""
		self.representations = representations

	def form(self, technique: EntityLocatingTechnique) -> str:
		"""
		Retrieves the representation of the answer with the given entity locating technique.

		:pattern technique: The technique with which to locate named entities.
		:raises ValueError: When the entity locating technique is not supported for this answer.
		:returns: The SPARQL answer, represented using the requested entity locating technique.
		"""
		try:
			return self.representations[technique]
		except KeyError:
			raise ValueError('[%s] Could not find the technique \'%s\'!' % (self.__class__.__name__, technique.value,))

	def __str__(self) -> str:
		"""
		Returns a string representation of the SPARQL answer.

		:returns: The string representation.
		"""
		return 'SPARQLAnswer(...)'

	def as_json_map(self) -> Dict[str, Any]:
		"""
		Represents the answer as a JSON-ready mapping.

		:returns: A mapping that is ready to be converted to JSON.
		"""
		return {tec.value: self.representations[tec] for tec in self.representations.keys()}


class QAPair:
	"""
	A combination of a question in one or more natural languages plus a SPARQL answer representation.
	"""

	def __init__(
			self,
			identifier: int,
			depth: int,
			q: NaturalLanguageQuestion,
			a: SPARQLAnswer,
			result: bool) -> None:
		"""
		Constructs the question-answer pair.

		:param identifier: The Compositional Freebase Questions (CFQ) ID of the QA pair.
		:param depth: The recursion depth of this QA pair.
		:param q: The natural language question.
		:param a: The SPARQL answer.
		:param result: Whether the result is affirmative or negative.
		"""
		self.identifier = identifier
		self.depth = depth
		self.q = q
		self.a = a
		self.result = result

	def __str__(self) -> str:
		"""
		Represents the question-answer pair in string form.

		:returns: The string representation.
		"""
		return 'QAPair(id=%d, depth=%d, q=%s, a=%s)' % (self.identifier, self.depth, str(self.q), str(self.a))

	def as_json_map(self) -> Dict[str, Any]:
		"""
		Yields the QA pair as a JSON-ready mapping.

		:returns: A mapping that is ready to be converted to JSON.
		"""
		return \
			{
				'id': self.identifier,
				'depth': self.depth,
				'q': self.q.as_json_map(),
				'a': self.a.as_json_map(),
				'result': self.result
			}


def _language_of_key(key: str) -> Language:
	"""
	Internal use. Given a JSON key, determines the language associated with said key.

	:returns: The key's language.
	"""
	mt = re.search('(_)[a-z]{2}$', key)
	if mt is None:
		return Language.ENGLISH
	code: str = mt.group().replace('_', '')  # an ISO-639-1 language code, such as `zh`
	for language in Language:
		if code == language.value:
			return language
	raise ValueError('[_language_of_key] Unidentifiable language for key \'%s\'!' % (key,))


def _add_to_representations(
		rep: Dict[Language, Dict[EntityLocatingTechnique, str]],
		language: Language,
		technique: EntityLocatingTechnique,
		value: str) -> None:
	"""
	Internal use. Adds the specified language-technique representation to the representations mapping.

	:param rep: The representations map. Maps from natural languages and entity locating techniques to representations.
	:param language: The language of the current representation.
	:param technique: The entity locating technique of the current representation.
	:param value: The current representation.
	"""
	if language not in rep:
		rep[language] = {}
	rep[language][technique] = value


def _natural_language_question_from_raw_representation(raw: JSONQAPair) -> NaturalLanguageQuestion:
	"""
	Internal use. Extracts from a JSON object a natural language question.

	:param raw: The raw JSON object to attempt to extract a natural language question from.
	:returns: The natural language question.
	"""
	rep: Dict[Language, Dict[EntityLocatingTechnique, str]] = {}
	for key, val in raw.items():
		# search through the JSON keys, and map these to entries in our Python wrapper object
		if re.search('(questionPatternModEntities)((_)[a-z]{2})?$', key):
			_add_to_representations(rep, _language_of_key(key), EntityLocatingTechnique.MOD_PATTERN_ENTITIES, val)
		elif re.search('(questionWithBrackets)((_)[a-z]{2})?$', key):
			_add_to_representations(rep, _language_of_key(key), EntityLocatingTechnique.WITH_BRACKETS, val)
	return NaturalLanguageQuestion(representations=rep)


def _sparql_answer_from_raw_representation(raw: JSONQAPair) -> SPARQLAnswer:
	"""
	Internal use. Extracts from a JSON object a SPARQL answer.

	:param raw: The raw JSON object to attempt to extract a SPARQL answer from.
	:returns: The SPARQL answer.
	"""
	rep: Dict[EntityLocatingTechnique, str] = {}
	for key, val in raw.items():
		if re.search('(sparql)$', key):
			rep[EntityLocatingTechnique.NONE] = val
		elif re.search('(sparqlPattern)$', key):
			rep[EntityLocatingTechnique.PATTERN] = val
		elif re.search('(sparqlPatternModEntities)$', key):
			rep[EntityLocatingTechnique.MOD_PATTERN_ENTITIES] = val
	return SPARQLAnswer(representations=rep)


def _qa_pair_from_raw_representation(raw: JSONQAPair) -> QAPair:
	"""
	Internal use. Extracts from a JSON object a question-answer pair.

	:param raw: The raw JSON object to attempt to extract a question-answer pair from.
	:returns: The question-answer pair.
	"""
	return QAPair(
		identifier=raw['CFQquestionIdx'],
		depth=raw['recursionDepth'],
		q=_natural_language_question_from_raw_representation(raw),
		a=_sparql_answer_from_raw_representation(raw),
		result=raw['expectedResponse']
	)


def qa_pairs_from_json(location: Path) -> List[QAPair]:
	"""
	Given the location of a JSON file, attempts to yield QA pairs encoded in said JSON file.

	:param location: The location of the JSON file.
	:returns: Zero or more QA pairs, encoded in the JSON file.
	"""
	raw_qa_pairs: List[JSONQAPair]
	qa_pairs: List[QAPair] = []
	with open(location, 'r') as handle:
		raw_qa_pairs = json.load(handle)
	for qa_pair in raw_qa_pairs:
		qa_pairs.append(_qa_pair_from_raw_representation(qa_pair))
	return qa_pairs


if __name__ == '__main__':
	# count the number of characters of the JSON file's English questions, for use in the Google Translate API
	questions_answers = qa_pairs_from_json(Path(ORIGINAL_DATA_FILE))
	number_of_characters: int = 0
	for question_answer in questions_answers:
		number_of_characters += \
			len(question_answer.q.form(Language.ENGLISH, EntityLocatingTechnique.WITH_BRACKETS, question_mark=True))
	print('Total number of characters to process: %d.' % (number_of_characters,))
