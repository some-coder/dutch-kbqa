"""
A class that serves as an interface to the LC Quad 2.0 KBQA dataset.
"""


import json
import os
import requests as req

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from pre_processing.answer import Answer, AnswerForm, StringAnswer
from pre_processing.language import FormalLanguage, NaturalLanguage
from pre_processing.question import Question, QuestionForm, StringQuestion
from pre_processing.question_answer_pair import RawQAPair, Metadata
from pre_processing.dataset.dataset import Dataset, RawDataset

from utility.typing import HTTPAddress


class LCQuAD(Dataset):

	QUESTION_KEYS: Dict[str, QuestionForm] = \
		{
			'NNQT_question': QuestionForm.BRACKETED,
			'question': QuestionForm.NORMAL,
			'paraphrased_question': QuestionForm.PARAPHRASED
		}
	ANSWER_KEYS: Dict[str, AnswerForm] = \
		{
			'sparql_wikidata': AnswerForm.WIKIDATA_NORMAL,
			'sparql_dbpedia18': AnswerForm.DBPEDIA_18_NORMAL
		}
	METADATA_KEYS: Tuple[str, ...] = \
		(
			'uid', 'subgraph', 'template_index', 'template', 'template_id', 'answer'
		)

	def __init__(self, dataset_locations: Optional[Tuple[Union[Path, HTTPAddress], ...]] = None) -> None:
		super().__init__(dataset_locations)
		res = req.get(self._dataset_locations[1])
		if res.status_code != 200:
			raise RuntimeError('Response code for URL \'%s\' was not \'200: OK\'!' % (self._dataset_locations[1],))
		self.num_test: int = len(json.loads(res.content))

	@property
	def _default_dataset_locations(self) -> Tuple[Union[Path, HTTPAddress], ...]:
		return \
			HTTPAddress('https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/train.json'), \
			HTTPAddress('https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/test.json')

	def _obtain_dataset(self) -> None:
		joined: RawDataset = RawDataset(tuple())
		ds_loc: HTTPAddress
		for ds_loc in self._dataset_locations:
			res = req.get(ds_loc)
			if res.status_code != 200:
				raise RuntimeError('Response code for URL \'%s\' was not \'200: OK\'!' % (ds_loc,))
			joined += tuple(json.loads(res.content))
		if not os.path.exists(self._dataset_save_file()):
			os.makedirs(Path(Dataset.DATASET_SAVE_DIRECTORY, self.__class__.__name__.lower()))
		with open(self._dataset_save_file(), 'x') as handle:
			json.dump(joined, handle)

	def _obtained_dataset(self) -> RawDataset:
		loc = Path(self._dataset_save_file())
		if self._dataset_is_already_stored():
			print('Dataset already stored! Retrieving it...')
			with open(loc, 'r') as handle:
				return json.load(handle)
		else:
			self._obtain_dataset()
			return self._obtained_dataset()  # recursive call, but should be OK

	def _question_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Question:
		forms: Dict[QuestionForm, Dict[NaturalLanguage, StringQuestion]] = {}
		for key, form in LCQuAD.QUESTION_KEYS.items():
			str_q: StringQuestion = raw_qa_pair[key]
			forms[form] = {}
			forms[form][NaturalLanguage.ENGLISH] = str_q
		return Question(forms)

	def _answer_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Answer:
		forms: Dict[AnswerForm, Dict[FormalLanguage, StringAnswer]] = {}
		for key, form in LCQuAD.ANSWER_KEYS.items():
			str_a: StringAnswer = raw_qa_pair[key]
			forms[form] = {}
			forms[form][FormalLanguage.SPARQL] = str_a
		return Answer(forms)

	def _metadata_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Metadata:
		metadata: Metadata = Metadata({})
		for key in LCQuAD.METADATA_KEYS:
			metadata[key] = raw_qa_pair[key]
		return metadata

	def questions_for_translation(self) -> Tuple[StringQuestion, ...]:
		return tuple(qa.q.in_form(QuestionForm.BRACKETED, NaturalLanguage.ENGLISH) for qa in self.qa_pairs)
