"""
Contains classes and methods for representing KBQA questions.
"""


from __future__ import annotations

from enum import Enum
from typing import Dict, NewType, Optional, Set, Tuple
from utility.typing import WikiDataSymbol

from utility.language import NaturalLanguage


StringQuestion = NewType('StringQuestion', str)


class QuestionForm(Enum):
	"""
	A way to represent a question.
	"""
	NORMAL = 'normal'
	BRACKETED = 'bracketed'
	BRACKETED_ENTITIES = 'bracketed-entities'
	BRACKETED_ENTITIES_RELATIONS = 'bracketed-entities-relations'
	PATTERNS = 'patterns'
	PATTERNS_ENTITIES = 'patterns-entities'
	PATTERNS_ENTITIES_RELATIONS = 'patterns-entities-relations'
	PARAPHRASED = 'paraphrased'


ENTITY_BRACKETS: Tuple[str, str] = ('(', ')')
RELATION_BRACKETS: Tuple[str, str] = ('((', '))')


QuestionFormMap = Dict[QuestionForm, Dict[NaturalLanguage, StringQuestion]]
QuestionForms = Tuple[QuestionForm, ...]


class Question:
	"""
	A KBQA question that can be represented in multiple forms.
	"""

	def __init__(self, forms: QuestionFormMap) -> None:
		self._forms = forms
		self._bracket_resolver: Optional[Dict[str, WikiDataSymbol]] = None
		self._pattern_resolver: Optional[Dict[str, WikiDataSymbol]] = None

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

	def initialise_bracket_resolver(self, br: Dict[str, WikiDataSymbol]) -> None:
		self._bracket_resolver = br

	def initialise_pattern_resolver(self, pr: Dict[str, WikiDataSymbol]) -> None:
		self._pattern_resolver = pr

	def resolve_bracket(self, bracket: str) -> WikiDataSymbol:
		return self._bracket_resolver[bracket]

	def resolve_pattern(self, pattern: str) -> WikiDataSymbol:
		return self._pattern_resolver[pattern]
