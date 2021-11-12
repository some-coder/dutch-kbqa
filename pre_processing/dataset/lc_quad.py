"""
A class that serves as an interface to the LC Quad 2.0 KBQA dataset.
"""


from __future__ import annotations

import json
import numpy as np
import os
import re
import requests as req

import utility.match as um

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypedDict, Union, cast

from pre_processing.answer import Answer, AnswerForm, StringAnswer, AnswerFormMap, AnswerForms
from pre_processing.translation import _confirm_access, _configure_credentials, translate_text
from pre_processing.question import ENTITY_BRACKETS, RELATION_BRACKETS, Question, QuestionForm, StringQuestion, \
	QuestionFormMap, QuestionForms
from pre_processing.question_answer_pair import RawQAPair, Metadata, QuestionKey, AnswerKey, MetadataKey, QAPair
from pre_processing.dataset.dataset import Dataset, RawDataset, bracketed_forms, patterned_forms

from utility.language import NaturalLanguage, FormalLanguage
from utility.typing import HTTPAddress, WikiDataSymbol
from utility.wikidata import WikiDataType, symbol_labels


ADDENDA_PATH = Path('resources/datasets/lcquad/addenda.json')
TRANSLATIONS_PATH = Path('resources/datasets/lcquad/translations.json')


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


class LCQuADTranslation(TypedDict):
	language: NaturalLanguage  # the language of the translation
	sentence: str  # the translated entity- or entity-relation-pattern sentence
	wd_symbols: Dict[WikiDataSymbol, str]  # a mapping from WikiData symbols (Q..., P...) to translated lexemes


class LCQuADTranslationRaw(TypedDict):  # as ``LCQuADTranslation``, but serializable
	language: str
	sentence: str
	wd_symbols: Dict[WikiDataSymbol, str]


Addenda = Dict[int, AddendaEntry]


class LCQuAD(Dataset):

	class QuestionKeysType(TypedDict):
		NNQT_question: QuestionForm
		question: QuestionForm
		paraphrased_question: QuestionForm

	class AnswerKeysType(TypedDict):
		sparql_wikidata: AnswerForm
		sparql_dbpedia18: AnswerForm

	QUESTION_KEYS: QuestionKeysType = \
		{
			'NNQT_question': QuestionForm.BRACKETED,
			'question': QuestionForm.NORMAL,
			'paraphrased_question': QuestionForm.PARAPHRASED
		}

	ANSWER_KEYS: LCQuAD.AnswerKeysType = \
		{
			'sparql_wikidata': AnswerForm.WIKIDATA_NORMAL,
			'sparql_dbpedia18': AnswerForm.DBPEDIA_18_NORMAL
		}

	METADATA_KEYS: Tuple[MetadataKey, ...] = \
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

	MISSING_SYMBOLS: Tuple[str] = ('n/a', 'None')

	def __init__(self, dataset_locations: Optional[Tuple[Union[Path, HTTPAddress], ...]] = None) -> None:
		self._addenda: Optional[Addenda] = None
		self._wd_symbols_to_q_p: Dict[int, Dict[WikiDataSymbol, str]] = {}
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

	@staticmethod
	def _pre_processed_dataset(raw_ds: RawDataset) -> RawDataset:
		for entry in raw_ds:
			raw_question: str = entry['question']
			if raw_question is None or raw_question in LCQuAD.MISSING_SYMBOLS:
				entry['question'] = None
				continue
			raw_question = re.sub('[(){}"]+', '', raw_question)  # remove any redundant parentheses
			raw_question = re.sub('[? ]+$', '', raw_question)  # remove question marks (we can re-add them later)
			entry['question'] = raw_question
		return raw_ds

	def _pre_process_addenda(self) -> None:
		"""
		Pre-process the addenda file for integration with the complete QA data.

		Specifically for LC-QuAD 2.0, we remove any trailing question marks; these nearly always are a remnant of
		matching the string from the original question, where the question mark was the last character of the
		question.
		"""
		for add_id in self._addenda.keys():
			link_keys = tuple(self._addenda[add_id]['links'].keys())
			for link_key in link_keys:
				wd_symbol: WikiDataSymbol = self._addenda[add_id]['links'][link_key]
				del self._addenda[add_id]['links'][link_key]
				revised_key: str = re.sub('(\\?)+$', '', link_key)
				self._addenda[add_id]['links'][revised_key] = wd_symbol

	def _obtained_dataset(self) -> RawDataset:
		loc = Path(self._dataset_save_file())
		if os.path.exists(self._default_addenda_location):
			print('[%s] Addenda file found! Retrieving it...' % (self.__class__.__name__,))
			with open(self._default_addenda_location, 'r') as handle:
				self._addenda = {int(kv[0]): kv[1] for kv in json.load(handle).items()}
				self._pre_process_addenda()
		if self._dataset_is_already_stored():
			print('[%s] Dataset already stored! Retrieving it...' % (self.__class__.__name__,))
			with open(loc, 'r') as handle:
				return self._pre_processed_dataset(json.load(handle))
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
	def _serialisation_ready_bracket_resolver(qa_pair: QAPair) -> AddendaEntry:
		br = LCQuAD._bracket_resolver_from_qa_pair(qa_pair)
		return {'links': br[0], 'quality': br[1].value}

	def _qa_pair_translation(self, qa_pair: QAPair, language: NaturalLanguage) -> LCQuADTranslationRaw:
		for f in (QuestionForm.PATTERNS_ENTITIES_RELATIONS, QuestionForm.PATTERNS_ENTITIES, QuestionForm.NORMAL):
			try:
				sen = qa_pair.q.in_form(f, NaturalLanguage.ENGLISH)
				translated = translate_text(sen, language)
				link_translations: Dict[WikiDataSymbol, str] = {}
				uid: int = int(qa_pair.metadata['uid'])
				for link_key, link_val in self._addenda[uid]['links'].items():
					link_translations[WikiDataSymbol(self._wd_symbols_to_q_p[uid][link_val])] = translate_text(link_key, language)
				return {'language': language.value, 'sentence': translated, 'wd_symbols': link_translations}
			except KeyError:
				continue  # try another form
		raise ValueError(
			'[_qa_pair_translation] Question %d does not have any suitable form!' % (int(qa_pair.metadata['uid']),))

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
		if q_or_a == Question:
			s = raw['question']
		else:
			s = raw['sparql_wikidata']
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

	def _update_wd_symbols_to_q_p(
			self,
			uid: int,
			wd_symbol: WikiDataSymbol,
			count: int) -> None:
		letter: str = wd_symbol[0]  # get the Q- or P-symbol
		if uid not in self._wd_symbols_to_q_p.keys():
			self._wd_symbols_to_q_p[uid] = {}
		self._wd_symbols_to_q_p[uid][wd_symbol] = '%s%d' % (letter, count)

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
					self._update_wd_symbols_to_q_p(int(raw['uid']), link_val, q_p_counts[wd_type])
					q_p_counts[wd_type] += 1  # only increment Q- and P-counts when we actually substitute
				d[key][sub_key] = StringQuestion(substituted) if q_or_a == Question else StringAnswer(substituted)
		return d

	def _question_or_answer_from_raw_qa_pair(
			self,
			raw_qa_pair: RawQAPair,
			q_or_a: Union[Type[Question], Type[Answer]]) -> Union[Question, Answer]:
		forms: Union[QuestionFormMap, AnswerFormMap] = {}
		key_items = LCQuAD.QUESTION_KEYS if q_or_a == Question else LCQuAD.ANSWER_KEYS
		for key, form in key_items.items():
			key: Union[QuestionKey, AnswerKey]  # guaranteed
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
		return {key: raw_qa_pair[key] for key in LCQuAD.METADATA_KEYS}

	def questions_for_translation(self) -> Tuple[StringQuestion, ...]:
		"""
		Deprecated. Yields the questions to translate.

		:return: The questions for use in translation.
		"""
		return tuple(qa.q.in_form(QuestionForm.BRACKETED, NaturalLanguage.ENGLISH) for qa in self.qa_pairs)

	def _computed_per_qa_pair(
			self,
			lc_quad_f: Callable,
			index_range: Tuple[int, int],
			extra_args: Optional[Dict[str, Any]] = None,
			update_progress: bool = True) -> Dict[int, Any]:
		"""
		Computes the function ``lc_quad_f`` per QA pair in the interval ``index_range``.

		:param lc_quad_f: The function to apply to all QA pairs in the index range.
		:param index_range: The range to select QA pairs in. Inclusive, exclusive.
		:param extra_args: Optional. Any extra arguments to supply to ``lc_quad_f``, besides the QA pair.
		:param update_progress: Optional. Whether to keep informed about progress. Defaults to ``True``.
		:return: A mapping from QA pair UIDs to ``lc_quad_f`` outputs. The mapping may be empty.
		"""
		d: Dict[int, Any] = {}
		if update_progress:
			print('Computing method output per question-answer pair...')
		for index in range(*index_range):
			if update_progress:
				print(
					'\t%3d / %3d (%6.2f%%)' %
					(
						index + 1 - index_range[0],
						index_range[1] - index_range[0],
						((index + 1 - index_range[0]) / (index_range[1] - index_range[0])) * 1e2
					))
			qa_pair: QAPair = self.qa_pairs[index]
			d[qa_pair.metadata['uid']] = lc_quad_f(self, qa_pair, **extra_args)
		return d

	def question_addenda(
			self,
			index_range: Tuple[int, int],
			update_progress: bool = True) -> Dict[int, AddendaEntry]:
		return self._computed_per_qa_pair(LCQuAD._bracket_resolver_from_qa_pair, index_range, {}, update_progress)

	def question_translations(
			self,
			language: NaturalLanguage,
			index_range: Tuple[int, int],
			update_progress: bool = True) -> Dict[int, LCQuADTranslation]:
		return self._computed_per_qa_pair(
			LCQuAD._qa_pair_translation,
			index_range,
			{
				'language': language
			},
			update_progress
		)


def _create_lc_quad_qa_pair_based_file(
		f: Callable,
		args: Dict[str, Any],
		path: Path,
		qa_pair_range: Optional[Tuple[int, int]] = None,
		update_progress: bool = True) -> None:
	lcq = LCQuAD()
	print('CREATED THE DATASET')
	if 'self' not in args.keys():
		args['self'] = lcq
	qa_ran: Tuple[int, int] = (0, len(lcq.qa_pairs)) if qa_pair_range is None else qa_pair_range
	previous: Dict[Any, Any] = {}
	if os.path.exists(path):
		with open(path, 'r') as handle:
			previous = json.load(handle)
	ran: Tuple[int, ...] = tuple(np.arange(*qa_ran))
	print('RANGE:')
	print(ran)
	for i in range(len(ran) - 1):
		print('(%d)' % (i,))
		start: int = ran[i]
		end: int = ran[i + 1]
		if update_progress:
			print('QA PAIRS FROM %4d TO %4d' % (start, end))
		current = f(**args)
		with open(path, 'w') as handle:
			previous.update(current)
			json.dump(previous, handle)  # save the JSON to disk


def create_lc_quad_addenda(qa_pair_range: Tuple[int, int], update_progress: bool = True) -> None:
	"""
	Generates addenda for the LC-QuAD 2.0 dataset.

	An 'Addendum' to a question-answer pair is a mapping between lexemes and WikiData Q- or P-values.
	These links can later be used by the LC-QuAD 2.0 dataset Python class to automatically generate
	some extra representations of the QA pair that were not available in the original pair, such
	as an entity-and-relation-bracketed or -masked representation.

	The addenda are saved to disk; the LC-QuAD 2.0 dataset Python class, once instantiated, will know
	where to look for this file.

	:param qa_pair_range: The indices of the QA pairs to create addenda for. Inclusive, exclusive.
	:param update_progress: Whether to notify the user of progress. Defaults to ``True``.
	"""
	_create_lc_quad_qa_pair_based_file(
		LCQuAD.question_addenda,
		{
			'index_range': qa_pair_range,
			'update_progress': update_progress
		},
		ADDENDA_PATH,
		qa_pair_range,
		update_progress
	)


def create_lc_quad_translations(
		language: NaturalLanguage,
		qa_pair_range: Tuple[int, int],
		update_progress: bool = True) -> None:
	"""
	Generates translations for the LC-QuAD 2.0 dataset.

	The translations are saved to disk. The next time an LC-QuAD 2.0 dataset Python class is instantiated,
	you can explicitly request said instantiation to load in the translations via a special method.

	In order to translate a QA pair, this method requires said pair to possess either an entity-pattern
	or an entity-relation-pattern representation. If this is not the case, the pair is silently skipped.

	TODO: Define the translation loading method in the ``LCQuAD`` class.

	:param language: The language to make translations for.
	:param qa_pair_range: The indices of the QA pairs to create translations for. Inclusive, exclusive.
	:param update_progress: Whether to notify the user of progress. Defaults to ``True``.
	"""
	_confirm_access()  # do not unnecessarily translate via Google's API
	_configure_credentials()
	_create_lc_quad_qa_pair_based_file(
		LCQuAD.question_translations,
		{
			'language': language,
			'index_range': qa_pair_range,
			'update_progress': update_progress
		},
		TRANSLATIONS_PATH,
		qa_pair_range,
		update_progress
	)
