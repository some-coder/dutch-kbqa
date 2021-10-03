"""
Contains classes and methods for representing KBQA questions.
"""


from __future__ import annotations

from enum import Enum
from typing import Dict, NewType, Set, Tuple

from pre_processing.language import NaturalLanguage


StringQuestion = NewType('StringQuestion', str)


class QuestionForm(Enum):
	"""
	A way to represent a question.
	"""
	NORMAL = 'normal'
	BRACKETED = 'bracketed'
	PATTERNS = 'patterns'
	PARAPHRASED = 'paraphrased'


class Question:
	"""
	A KBQA question that can be represented in multiple forms.
	"""

	def __init__(self, forms: Dict[QuestionForm, Dict[NaturalLanguage, StringQuestion]]) -> None:
		self._forms = forms

	def in_form(self, form: QuestionForm, language: NaturalLanguage) -> StringQuestion:
		return self._forms[form][language]

	@property
	def question_forms(self) -> Tuple[QuestionForm, ...]:
		return tuple(self._forms.keys())

	@property
	def natural_languages(self) -> Tuple[NaturalLanguage, ...]:
		s: Set[NaturalLanguage] = set()
		for qf in self.question_forms:
			for nl in self._forms[qf].keys():
				s.add(nl)
		return tuple(s)
