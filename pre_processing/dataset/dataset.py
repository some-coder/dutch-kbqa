"""
Stores a base class for making KBQA datasets.
"""


import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, NewType, Optional, Tuple, Type, Union

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

	DATASET_SAVE_DIRECTORY = Path('resources', 'datasets')

	def __init__(self, dataset_locations: Optional[Tuple[Union[Path, HTTPAddress], ...]] = None) -> None:
		if dataset_locations is None:
			self._dataset_locations = self._default_dataset_locations
		else:
			self._dataset_locations = dataset_locations
		raw = self._obtained_dataset()
		if len(raw) == 0:
			raise ValueError('The dataset does not contain any QA pairs!')
		self.qa_pairs = self._created_qa_pairs(raw)
		self.dom_ran: Dict[Domain, Range] = self._created_dom_ran(raw[0])

	def _dataset_save_file(self) -> Path:
		return Path(Dataset.DATASET_SAVE_DIRECTORY, self.__class__.__name__.lower(), 'data.json')

	def _dataset_is_already_stored(self) -> bool:
		return os.path.exists(self._dataset_save_file())

	@property
	@abstractmethod
	def _default_dataset_locations(self) -> Tuple[Union[Path, HTTPAddress], ...]:
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
