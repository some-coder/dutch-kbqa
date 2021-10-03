"""
A class for representing a KBQA question-answer pair.
"""


from typing import Any, Dict, NewType

from pre_processing.question import Question
from pre_processing.answer import Answer


Metadata = NewType('Metadata', Dict[str, Any])
RawQAPair = NewType('RawQAType', Dict[str, Any])


class QAPair:

	def __init__(self, q: Question, a: Answer, metadata: Metadata) -> None:
		self.q = q
		self.a = a
		self.metadata = metadata
