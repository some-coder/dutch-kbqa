"""
Contains classes and methods for representing KBQA answers.
"""


from enum import Enum
from typing import Dict, NewType, Tuple

from pre_processing.language import FormalLanguage


StringAnswer = NewType('StringAnswer', str)


class AnswerForm(Enum):
	"""
	A way to represent an answer.
	"""
	NORMAL = 'normal'
	PATTERNS = 'patterns'


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
		return tuple({self._forms[a].keys() for a in self.answer_forms})
