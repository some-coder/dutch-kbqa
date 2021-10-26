"""
A class that serves as an interface to the LC Quad 2.0 KBQA dataset.
"""

import json
import os
import re
import requests as req

import utility.match as um

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypedDict, Union, cast

from pre_processing.answer import Answer, AnswerForm, StringAnswer, AnswerFormMap, AnswerForms
from pre_processing.question import ENTITY_BRACKETS, RELATION_BRACKETS, Question, QuestionForm, StringQuestion, \
	QuestionFormMap, QuestionForms
from pre_processing.question_answer_pair import RawQAPair, Metadata, QAPair
from pre_processing.dataset.dataset import Dataset, RawDataset, bracketed_forms, patterned_forms

from utility.language import NaturalLanguage, FormalLanguage
from utility.typing import HTTPAddress, WikiDataSymbol
from utility.wikidata import WikiDataType, symbol_labels


class QualityGroup(Enum):
	NONE = 'none'
	Q = 'q'
	Q_AND_P = 'q-and-p'


class AddendaEntry(TypedDict):
	links: Dict[str, WikiDataSymbol]
	quality: str  # the string form of a `QualityGroup`


@dataclass
class ReplaceInstruction:
	pat: str  # the pattern to replace
	rep: str  # the replacement for the pattern
	use_reg_exp: bool  # whether to use RegExp (`True`) or plain `replace` (`False`) for replacement


Addenda = Dict[int, AddendaEntry]


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

	# Note: order may matter, so be sure to place instructions in a sequence that makes sense!
	QUESTION_NORMAL_REPLACE_INSTRUCTIONS: Tuple[ReplaceInstruction, ...] = \
		(
			ReplaceInstruction('\\', '', False),
		)
	QUESTION_ADDENDA_REPLACE_INSTRUCTIONS: Tuple[ReplaceInstruction, ...] = \
		(
			ReplaceInstruction('\\', '', False),
			ReplaceInstruction('(of )', '', True)
		) + \
		tuple(
			ReplaceInstruction('%s' % (sym,), '\\%s' % (sym,), False) for sym in
			ENTITY_BRACKETS + RELATION_BRACKETS + ('+', '[', ']')
		)
	ANSWER_NORMAL_REPLACE_INSTRUCTIONS: Tuple[ReplaceInstruction, ...] = \
		(
			ReplaceInstruction('(wd)(t)?(:)', '', True),
		)

	def __init__(self, dataset_locations: Optional[Tuple[Union[Path, HTTPAddress], ...]] = None) -> None:
		self._addenda: Optional[Addenda] = None
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
	def _should_replace_with_addenda(
			q_or_a: Union[Type[Question], Type[Answer]],
			form: Union[QuestionForm, AnswerForm],
			wd_type: WikiDataType) -> bool:
		if wd_type == WikiDataType.ENTITY:
			return True
		elif wd_type == WikiDataType.RELATION:
			return \
				(
					q_or_a == Question and
					form in (
						QuestionForm.BRACKETED_ENTITIES_RELATIONS,
						QuestionForm.PATTERNS_ENTITIES_RELATIONS)
				) or \
				(
					q_or_a == Answer and
					form in (
						AnswerForm.WIKIDATA_BRACKETED_ENTITIES_RELATIONS,
						AnswerForm.WIKIDATA_PATTERNS_ENTITIES_RELATIONS)
				)
		return False

	@staticmethod
	def _addenda_replacement(
			s: str,
			q_or_a: Union[Type[Question], Type[Answer]],
			form: Union[QuestionForm, AnswerForm],
			wd_type: WikiDataType,
			i: int) -> str:
		if form in bracketed_forms() and LCQuAD._should_replace_with_addenda(q_or_a, form, wd_type):
			breaks: Tuple[str, str] = ENTITY_BRACKETS if wd_type == WikiDataType.ENTITY else RELATION_BRACKETS
			return '%s%s%s' % (breaks[0], s, breaks[1])
		elif form in patterned_forms() and LCQuAD._should_replace_with_addenda(q_or_a, form, wd_type):
			return '%s%d' % (wd_type.value, i)
		return s  # this form does not demand any replacements via addenda

	@staticmethod
	def _massaged_addenda_string(s: str, q_or_a: Union[Type[Question], Type[Answer]]) -> str:
		if q_or_a == Answer:
			return s  # Q- and P-values are already 'neat'.
		else:
			for ins in LCQuAD.QUESTION_ADDENDA_REPLACE_INSTRUCTIONS:
				s = re.sub(ins.pat, ins.rep, s) if ins.use_reg_exp else s.replace(ins.pat, ins.rep)
			return s

	@staticmethod
	def _normal_form_with_replacements(
			raw: RawQAPair,
			q_or_a: Union[Type[Question], Type[Answer]]) -> Union[StringQuestion, StringAnswer]:
		s = raw['question' if q_or_a == Question else 'sparql_wikidata']
		rep_ins: Tuple[ReplaceInstruction, ...] = \
			LCQuAD.QUESTION_NORMAL_REPLACE_INSTRUCTIONS if q_or_a == Question else LCQuAD.ANSWER_NORMAL_REPLACE_INSTRUCTIONS
		for ins in rep_ins:
			s = re.sub(ins.pat, ins.rep, s) if ins.use_reg_exp else s.replace(ins.pat, ins.rep)
		return StringQuestion(s) if q_or_a == Question else StringAnswer(s)

	@staticmethod
	def _should_substitute(
			link_item: Tuple[str, WikiDataSymbol],
			q_or_a: Union[Type[Question], Type[Answer]],
			form: Union[QuestionForm, AnswerForm]) -> bool:
		return not (
			q_or_a == Answer and
			form in (AnswerForm.WIKIDATA_BRACKETED_ENTITIES, AnswerForm.WIKIDATA_PATTERNS_ENTITIES) and
			link_item[1][0] == WikiDataType.RELATION.value
		)

	def _question_or_answer_addenda(
			self,
			raw: RawQAPair,
			q_or_a: Union[Type[Question], Type[Answer]]) -> Union[QuestionFormMap, AnswerFormMap]:
		d: Union[QuestionFormMap, AnswerFormMap] = {}
		add: AddendaEntry
		if self._addenda is None:
			raise RuntimeError('[_question_or_answer_addenda] Addenda not found!')
		else:
			add = cast(AddendaEntry, self._addenda[int(raw['uid'])])
		keys: Union[QuestionForms, AnswerForms] = tuple()
		if add['quality'] in (QualityGroup.Q.value, QualityGroup.Q_AND_P.value):
			keys += \
				(QuestionForm.BRACKETED_ENTITIES, QuestionForm.PATTERNS_ENTITIES) if q_or_a == Question else \
				(AnswerForm.WIKIDATA_BRACKETED_ENTITIES, AnswerForm.WIKIDATA_PATTERNS_ENTITIES)
		if add['quality'] == QualityGroup.Q_AND_P.value:
			keys += \
				(QuestionForm.BRACKETED_ENTITIES_RELATIONS, QuestionForm.PATTERNS_ENTITIES_RELATIONS) if q_or_a == Question else \
				(AnswerForm.WIKIDATA_BRACKETED_ENTITIES_RELATIONS, AnswerForm.WIKIDATA_PATTERNS_ENTITIES_RELATIONS)
		for key in keys:
			# go over all question or answer forms
			q_p_counts: Dict[WikiDataType, int] = {wdt: 0 for wdt in (WikiDataType.ENTITY, WikiDataType.RELATION)}
			sub_key: Union[NaturalLanguage, FormalLanguage] = \
				NaturalLanguage.ENGLISH if q_or_a == Question else FormalLanguage.SPARQL
			d[key] = {}
			d[key][sub_key] = LCQuAD._normal_form_with_replacements(raw, q_or_a)
			for link_key, link_val in add['links'].items():
				# replace the original question or answer piece-by-piece
				massaged: str = LCQuAD._massaged_addenda_string(link_key, Type[Question])
				wd_type: WikiDataType = WikiDataType.ENTITY if link_val[0] == WikiDataType.ENTITY.value else WikiDataType.RELATION
				replacement = LCQuAD._addenda_replacement(massaged, q_or_a, key, wd_type, q_p_counts[wd_type]).replace('\\', '')
				substituted = d[key][sub_key]  # simply the normal question or answer, in case of no substitution
				if LCQuAD._should_substitute((link_key, link_val), q_or_a, key):
					substituted = re.sub('(%s)' % (massaged if q_or_a == Question else link_val,), replacement, substituted)
				d[key][sub_key] = StringQuestion(substituted) if q_or_a == Question else StringAnswer(substituted)
				q_p_counts[wd_type] += 1
		return d

	def _question_or_answer_from_raw_qa_pair(
			self,
			raw_qa_pair: RawQAPair,
			q_or_a: Union[Type[Question], Type[Answer]]) -> Union[Question, Answer]:
		forms: Union[QuestionFormMap, AnswerFormMap] = {}
		key_items: Union[Dict[str, QuestionForm], Dict[str, AnswerForm]] = \
			LCQuAD.QUESTION_KEYS if q_or_a == Question else LCQuAD.ANSWER_KEYS
		for key, form in key_items.items():
			s: Union[StringQuestion, StringAnswer] = \
				StringQuestion(raw_qa_pair[key]) if q_or_a == Question else StringAnswer(raw_qa_pair[key])
			forms[form] = {NaturalLanguage.ENGLISH: s} if q_or_a == Question else {FormalLanguage.SPARQL: s}
		if self._addenda is not None and int(raw_qa_pair['uid']) in self._addenda.keys():
			# retrieve additional addenda information, as it's available
			forms.update(self._question_or_answer_addenda(raw_qa_pair, q_or_a))
		return Question(forms) if q_or_a == Question else Answer(forms)

	def _question_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Question:
		return self._question_or_answer_from_raw_qa_pair(raw_qa_pair, Question)

	def _answer_from_raw_qa_pair(self, raw_qa_pair: RawQAPair) -> Answer:
		return self._question_or_answer_from_raw_qa_pair(raw_qa_pair, Answer)

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
