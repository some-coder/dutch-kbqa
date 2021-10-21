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
from pre_processing.question import ENTITY_BRACKETS, RELATION_BRACKETS, Question, QuestionForm, StringQuestion
from pre_processing.question_answer_pair import RawQAPair, Metadata, QAPair
from pre_processing.dataset.dataset import Dataset, RawDataset

from utility.language import NaturalLanguage, FormalLanguage
from utility.typing import HTTPAddress, WikiDataSymbol
from utility.wikidata import WIKIDATA_ENTITY, WIKIDATA_RELATION, symbol_labels


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
		self._addenda: Optional[Dict[int, Dict[str, Any]]] = None
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

	@property
	def _default_addenda_location(self) -> Path:
		return Path('resources', 'datasets', 'lcquad', 'addenda.json')

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
		if os.path.exists(self._default_addenda_location):
			print('[%s] Addenda file found! Retrieving it...' % (self.__class__.__name__,))
			with open(self._default_addenda_location, 'r') as handle:
				self._addenda = {int(kv[0]): kv[1] for kv in json.load(handle).items()}
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

	@staticmethod
	def _question_addenda_replacement(
			string: str,
			key: QuestionForm,
			typ: Union[WIKIDATA_ENTITY, WIKIDATA_RELATION],
			i: int) -> str:
		if key in (QuestionForm.BRACKETED_ENTITIES, QuestionForm.BRACKETED_ENTITIES_RELATIONS):
			if typ == WIKIDATA_ENTITY or \
					(typ == WIKIDATA_RELATION and key == QuestionForm.BRACKETED_ENTITIES_RELATIONS):
				brk: Tuple[str, str] = ENTITY_BRACKETS if typ == WIKIDATA_ENTITY else RELATION_BRACKETS
				return '%s%s%s' % (brk[0], string, brk[1])
			else:
				return '%s' % (string,)  # relations in bracket-entity matching: leave it alone
		elif key in (QuestionForm.PATTERNS_ENTITIES, QuestionForm.PATTERNS_ENTITIES_RELATIONS):
			if typ == WIKIDATA_ENTITY or (typ == WIKIDATA_RELATION and key == QuestionForm.PATTERNS_ENTITIES_RELATIONS):
				return '%s%d' % (typ, i)
			else:
				return '%s' % (string,)  # relations in pattern-entity matching: leave it alone
		else:
			raise ValueError('Unrecognised question form: \'%s\'.' % (key.value,))

	@staticmethod
	def _massaged_addenda_string(raw: str) -> str:
		rep = raw.replace('\\', '')
		rep = re.sub('(of )', '', rep)
		rep = rep.replace('%s' % (ENTITY_BRACKETS[0],), '\\%s' % (ENTITY_BRACKETS[0]))
		rep = rep.replace('%s' % (ENTITY_BRACKETS[1],), '\\%s' % (ENTITY_BRACKETS[1]))
		rep = rep.replace('%s' % (RELATION_BRACKETS[0],), '\\%s' % (RELATION_BRACKETS[0]))
		rep = rep.replace('%s' % (RELATION_BRACKETS[1],), '\\%s' % (RELATION_BRACKETS[1]))
		rep = rep.replace('+', '\\+')
		rep = rep.replace('[', '\\[').replace(']', '\\]')
		return rep

	def _question_addenda_from_raw_qa_pair(
			self,
			raw_qa_pair: RawQAPair) -> Dict[QuestionForm, Dict[NaturalLanguage, StringQuestion]]:
		add: Dict[str, Any] = self._addenda[int(raw_qa_pair['uid'])]
		d: Dict[QuestionForm, Dict[NaturalLanguage, StringQuestion]] = {}
		keys: Tuple[QuestionForm, ...] = tuple()
		if add['quality'] in (QualityGroup.Q.value, QualityGroup.Q_AND_P.value):
			keys += (QuestionForm.BRACKETED_ENTITIES, QuestionForm.PATTERNS_ENTITIES)
		if add['quality'] == QualityGroup.Q_AND_P.value:
			keys += (QuestionForm.BRACKETED_ENTITIES_RELATIONS, QuestionForm.PATTERNS_ENTITIES_RELATIONS)
		for key in keys:
			q_d_counts: Tuple[int, int] = (0, 0)
			if key not in d:
				d[key] = {}
			# start with the original question
			d[key][NaturalLanguage.ENGLISH] = raw_qa_pair['question'].replace('\\', '')
			for lnk_key, lnk_val in add['links'].items():
				msg: str = LCQuAD._massaged_addenda_string(lnk_key)
				if lnk_val[0] == WIKIDATA_ENTITY:
					d[key][NaturalLanguage.ENGLISH] = StringQuestion(re.sub(
						'(%s)' % (msg,),
						LCQuAD._question_addenda_replacement(
							msg, key, WIKIDATA_ENTITY, q_d_counts[0]).replace('\\', ''),
						d[key][NaturalLanguage.ENGLISH]))
					q_d_counts = (q_d_counts[0] + 1, q_d_counts[1])
				elif lnk_val[0] == WIKIDATA_RELATION:
					d[key][NaturalLanguage.ENGLISH] = StringQuestion(re.sub(
						'(%s)' % (msg,),
						LCQuAD._question_addenda_replacement(
							msg, key, WIKIDATA_RELATION, q_d_counts[1]).replace('\\', ''),
						d[key][NaturalLanguage.ENGLISH]
					))
					q_d_counts = (q_d_counts[0], q_d_counts[1] + 1)
		return d

	def _question_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Question:
		forms: Dict[QuestionForm, Dict[NaturalLanguage, StringQuestion]] = {}
		for key, form in LCQuAD.QUESTION_KEYS.items():
			str_q: StringQuestion = raw_qa_pair[key]
			forms[form] = {}
			forms[form][NaturalLanguage.ENGLISH] = str_q
		if self._addenda is not None and int(raw_qa_pair['uid']) in self._addenda.keys():
			# retrieve additional addenda information, as it's available
			forms.update(self._question_addenda_from_raw_qa_pair(raw_qa_pair))
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
