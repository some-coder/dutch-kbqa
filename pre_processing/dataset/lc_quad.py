"""
A class that serves as an interface to the LC Quad 2.0 KBQA dataset.
"""


import json
import os
import re
import requests as req

import utility.match as um

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from pre_processing.answer import Answer, AnswerForm, StringAnswer
from pre_processing.question import Question, QuestionForm, StringQuestion
from pre_processing.question_answer_pair import RawQAPair, Metadata, QAPair
from pre_processing.dataset.dataset import Dataset, RawDataset

from utility.language import NaturalLanguage, FormalLanguage
from utility.typing import HTTPAddress, WikiDataSymbol
from utility.wikidata import symbol_labels


class QualityGroup(Enum):
	NONE = 'none'
	Q = 'q'
	Q_AND_P = 'q-and-p'


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
			print('[%s] Dataset already stored! Retrieving it...' % (self.__class__.__name__,))
			with open(loc, 'r') as handle:
				return json.load(handle)
		else:
			self._obtain_dataset()
			return self._obtained_dataset()  # recursive call, but should be OK

	@staticmethod
	def _sparql_wikidata_symbols(sparql: str) -> Tuple[WikiDataSymbol, ...]:
		symbols: Tuple[str, ...] = tuple()
		for match in re.finditer('(wd)(t)?(:)[QP][0-9]+', sparql):
			symbols += re.sub('(wd)(t)?(:)', '', match.group()),  # remove the URL prefix
		return tuple(WikiDataSymbol(s) for s in set(symbols))  # make unique

	@staticmethod
	def _matched_sparql_wikidata_symbol(string: str, sym: WikiDataSymbol) -> Optional[str]:
		"""
		Finds the lexeme in ``string`` that matches one of the WikiData symbol's labels.

		:param string: The string to find a mention of ``sym`` in.
		:param sym: The WikiData symbol to find a mention for.
		:return: The mention.
		"""
		lowered_string = string.lower()
		for label in symbol_labels(sym, NaturalLanguage.ENGLISH, wait=True):
			m = um.match(label.lower(), lowered_string, threshold=8e-1)
			if m is None:
				continue  # this label doesn't fit; try the next one
			return string[m[0]:m[1]]
		return None

	@staticmethod
	def _bracket_resolver_from_qa_pair(qa_pair: QAPair) -> Tuple[Dict[str, WikiDataSymbol], QualityGroup]:
		q: str = qa_pair.q.in_form(QuestionForm.NORMAL, NaturalLanguage.ENGLISH)
		if q is None:
			return {}, QualityGroup.NONE  # question does not exist
		s: str = qa_pair.a.in_form(AnswerForm.WIKIDATA_NORMAL, FormalLanguage.SPARQL)
		d: Dict[str, WikiDataSymbol] = {}
		ws = LCQuAD._sparql_wikidata_symbols(s)
		all_ps_linked: bool = True
		for wikidata_symbol in ws:
			match = LCQuAD._matched_sparql_wikidata_symbol(q, wikidata_symbol)
			if match is not None:
				d[match] = wikidata_symbol
			elif wikidata_symbol[0] == 'Q':
				return {}, QualityGroup.NONE
			else:
				all_ps_linked = False
		return d, QualityGroup.Q_AND_P if all_ps_linked else QualityGroup.Q

	def _question_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Question:
		forms: Dict[QuestionForm, Dict[NaturalLanguage, StringQuestion]] = {}
		for key, form in LCQuAD.QUESTION_KEYS.items():
			str_q: StringQuestion = raw_qa_pair[key]
			forms[form] = {}
			forms[form][NaturalLanguage.ENGLISH] = str_q
		q = Question(forms)
		return q

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

	def question_addenda(self, index_range: Tuple[int, int]) -> Dict[int, Any]:
		d: Dict[int, Any] = {}
		print('DERIVING ADDENDA')
		for index in range(*index_range):
			print(
				'\t%3d / %3d (%6.2f%%)' %
				(
					index + 1 - index_range[0],
					index_range[1] - index_range[0],
					((index + 1 - index_range[0]) / (index_range[1] - index_range[0])) * 1e2
				))
			qa_pair: QAPair = self.qa_pairs[index]
			tup = LCQuAD._bracket_resolver_from_qa_pair(qa_pair)
			d[qa_pair.metadata['uid']] = {'links': tup[0], 'quality': tup[1].value}
		return d
