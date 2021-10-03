"""
Stores a base class for making KBQA datasets.
"""


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, NewType, Optional, Tuple, Type, Union

from utility.typing import HTTPAddress

from pre_processing.answer import Answer, AnswerForm
from pre_processing.language import FormalLanguage, NaturalLanguage
from pre_processing.question import Question, QuestionForm
from pre_processing.question_answer_pair import Metadata, RawQAPair, QAPair


Domain = NewType(
	'Domain',
	Union[Type[QuestionForm], Type[NaturalLanguage], Type[AnswerForm], Type[FormalLanguage], Type[Metadata]])
Range = NewType(
	'Range',
	Union[
		Tuple[QuestionForm, ...], Tuple[NaturalLanguage, ...], Tuple[AnswerForm, ...], Tuple[FormalLanguage, ...],
		Tuple[str, ...]
	])
RawDataset = NewType('RawDataset', Tuple[RawQAPair, ...])


class Dataset(ABC):
	"""
	An abstract base class that represents a KBQA pre_processing.
	"""

	DATASET_SAVE_LOCATION = Path('resources', 'datasets')

	def __init__(self, dataset_location: Optional[Union[Path, HTTPAddress]] = None) -> None:
		if dataset_location is None:
			self.dataset_location = self._default_dataset_location
		else:
			self._dataset_location = dataset_location
		raw = self._obtained_dataset()
		if len(raw) == 0:
			raise ValueError('The dataset does not contain any QA pairs!')
		self.qa_pairs = self._created_qa_pairs(raw)
		self.dom_ran: Dict[Domain, Range] = self._created_dom_ran(raw[0])

	@abstractmethod
	@property
	def _default_dataset_location(self) -> Union[Path, HTTPAddress]:
		pass

	@abstractmethod
	def _obtained_dataset(self) -> RawDataset:
		pass

	@abstractmethod
	def _question_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Question:
		pass

	@abstractmethod
	def _answer_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Answer:
		pass

	@abstractmethod
	def _metadata_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Metadata:
		pass

	def _created_qa_pair(self, raw_qa_pair: RawQAPair) -> QAPair:
		return QAPair(
			q=self._question_from_raw_qa_pair(raw_qa_pair),
			a=self._answer_from_raw_qa_pair(raw_qa_pair),
			metadata=self._metadata_from_raw_qa_pair(raw_qa_pair))

	def _created_qa_pairs(self, raw_dataset: RawDataset) -> Tuple[QAPair, ...]:
		return tuple(self._created_qa_pair(r) for r in raw_dataset)

	def _created_dom_ran(self, raw_qa_pair: RawQAPair) -> Dict[Domain, Range]:
		return {
			QuestionForm: self.qa_pairs[0].q.question_forms,
			NaturalLanguage: self.qa_pairs[0].q.natural_languages,
			AnswerForm: self.qa_pairs[0].a.answer_forms,
			FormalLanguage: self.qa_pairs[0].a.formal_languages,
			Metadata: self._metadata_from_raw_qa_pair(raw_qa_pair)
		}  # we trust that all QA pairs are similarly formed
