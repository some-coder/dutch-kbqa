"""
A class for representing a KBQA question-answer pair.
"""


from typing import List, Literal, TypedDict, Union

from pre_processing.question import Question
from pre_processing.answer import Answer


class Metadata(TypedDict):
	uid: int
	subgraph: Union[str, List]  # if a list, it's always the empty list
	template_index: int
	template: Union[str, List]  # again, if a list, it's the empty list
	template_id: str
	answer: List  # always the empty list


class RawQAPair(TypedDict):
	NNQT_question: str
	question: str
	paraphrased_question: str
	sparql_wikidata: str
	sparql_dbpedia18: str
	uid: int
	subgraph: Union[str, List]
	template_index: int
	template: str
	template_id: str
	answer: List


QuestionKey = \
	Union[
		Literal['NNQT_question'],
		Literal['question'],
		Literal['paraphrased_question']
	]
AnswerKey = \
	Union[
		Literal['sparql_wikidata'],
		Literal['sparql_dbpedia18']
	]
MetadataKey = \
	Union[
		Literal['uid'],
		Literal['subgraph'],
		Literal['template_index'],
		Literal['template'],
		Literal['template_id'],
		Literal['answer']
	]


class QAPair:

	def __init__(self, q: Question, a: Answer, metadata: Metadata) -> None:
		self.q = q
		self.a = a
		self.metadata = metadata
