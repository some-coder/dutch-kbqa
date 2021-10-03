"""
Contains classes and methods for representing KBQA answers.
"""


from enum import Enum
from typing import Dict, NewType, Set, Tuple

from pre_processing.language import FormalLanguage


StringAnswer = NewType('StringAnswer', str)


class AnswerForm(Enum):
	"""
	A way to represent an answer.
	"""
	WIKIDATA_NORMAL = 'wikidata-normal'
	WIKIDATA_PATTERNS = 'wikidata-patterns'
	DBPEDIA_18_NORMAL = 'dbpedia-18-normal'
	DBPEDIA_18_PATTERNS = 'dbpedia-18-patterns'


class Answer:
	"""
	A KBQA answer that can be represented in multiple forms.
	"""

	def __init__(self, forms: Dict[AnswerForm, Dict[FormalLanguage, StringAnswer]]) -> None:
		self._forms = forms

	def in_form(self, form: AnswerForm, language: FormalLanguage) -> StringAnswer:
		return self._forms[form][language]

	@property
	def answer_forms(self) -> Tuple[AnswerForm, ...]:
		return tuple(k for k in self._forms.keys())

	@property
	def formal_languages(self) -> Tuple[FormalLanguage, ...]:
		s: Set[FormalLanguage] = set()
		for af in self.answer_forms:
			for key in self._forms[af].keys():
				s.add(key)
		return tuple(s)