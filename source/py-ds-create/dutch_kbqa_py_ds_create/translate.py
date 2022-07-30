"""Symbols for translating texts using Google Cloud Translate."""

import six
import json
import os
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate
from pathlib import Path
from dutch_kbqa_py_ds_create.lc_quad_2_0 import LCQuADQAPair, \
                                                Split, \
												dataset_split, \
                                                question_type_for_translation
from dutch_kbqa_py_ds_create.utilities import NaturalLanguage, \
                                              overwritably_print
from typing import Union, List, Dict, Set


def translated_text(text: Union[str, six.binary_type],
                    tgt_language: NaturalLanguage,
					src_language: NaturalLanguage) -> str:
	"""Returns `text`, machine-translated into the desired `tgt_language`.

	If you encounter a `ValueError` when calling this function, it is likely
	that your `text` uses a text encoding different from UTF-8, such as
	ISO 8859-1.

	:param text: The text to translate.
	:param tgt_language: The language to translate `text` into.
	:param src_language: The language in which `text` is written.
	:returns: `text`, machine-translated into `tgt_language`.
	:raises: `ValueError` if `text` is binary but not properly encoded as
		UTF-8.
	"""
	client = translate.Client()
	if isinstance(text, six.binary_type):
		try:	
			text = text.decode(encoding='utf-8')
		except UnicodeDecodeError:
			raise ValueError('`text` is binary but not (correctly) UTF-8 ' +
			                 'encoded.')
	result = client.translate(text,
	                          target_language=tgt_language.value,
	                          source_language=src_language.value)
	return result['translatedText']


TranslatedLCQuADQuestions = Dict[str, str]


def ds_split_uids(ds_split: List[LCQuADQAPair]) -> Set[int]:
	"""Returns the UIDs of the question-answer pairs in an LC-QuAD 2.0 dataset
	split.

	:param ds_split: The LC-QuAD 2.0 dataset split to return UIDs of.
	:returns: The UIDs.	
	"""
	return set(pair['uid'] for pair in ds_split)


def trl_questions_uids(trl_questions: TranslatedLCQuADQuestions) -> Set[int]:
	"""Returns the UIDs of the questions in a translated LC-QuAD 2.0 dataset
	split.
	
	:param trl_questions: A mapping from stringified UIDs to translated
		questions. Already-translated questions. May be incomplete or even
		empty.
	:returns: The questions' UIDs.	
	"""
	return set(int(key) for key in trl_questions.keys())


def questions_still_to_translate(ds_split: List[LCQuADQAPair],
                                 trl_questions: TranslatedLCQuADQuestions) -> Set[int]:
	"""Returns the UIDs of questions that still need to be translated.

	:param ds_split: The LC-QuAD 2.0 dataset split to translate questions of.
	:param trl_questions: A mapping from stringified UIDs to translated
		questions. Already-translated questions. May be incomplete or even
		empty.
	:returns: A set of UID. These are the UIDs of pairs from `ds_split` that 
		still need to be translated.
	"""	
	all_uids = ds_split_uids(ds_split)
	return all_uids.difference(trl_questions_uids(trl_questions))


def translate_dataset_split_questions(ds_split: List[LCQuADQAPair],
                                      trl_questions: TranslatedLCQuADQuestions,
							          language: NaturalLanguage,
									  question_uids: Set[int]) -> None:
	"""Translates questions from LC-QuAD 2.0 dataset split `ds_split` into
	`language`.

	This function translates questions with UIDs from `question_uids` in
	`ds_split` and places them into `trl_questions`. It is your responsibility
	to maintain `trl_questions`.

	:param ds_split: The LC-QuAD 2.0 dataset split to translate questions of.
	:param trl_questions: A mapping from stringified UIDs to translated
		questions. Already-translated questions. May be incomplete or even
		empty.
	:param language: The language to translate into.
	:param question_uids: The UIDs of the questions to translate.
	"""
	assert question_uids.issubset(ds_split_uids(ds_split))
	assert question_uids.isdisjoint(trl_questions_uids(trl_questions))
	uids_to_qa_pairs = {pair['uid']: pair for pair in ds_split}
	for uid in question_uids:
		pair = uids_to_qa_pairs[uid]
		question_type = question_type_for_translation(pair)
		text = pair[question_type]
		assert isinstance(text, str)
		# Note: LC-QuAD 2.0 is written in English, warranting us setting the
		# `src_language` to `NaturalLanguage.ENGLISH` without user 
		# intervention.
		trl = translated_text(text,
		                      tgt_language=language,
							  src_language=NaturalLanguage.ENGLISH)
		trl_questions[str(uid)] = trl


def trl_questions_from_disk(file: Path) -> TranslatedLCQuADQuestions:
	"""Returns the translated LC-QuAD 2.0 dataset split loaded from disk.
	
	:param file: The file to which translated questions are saved.
	:returns: The translated questions. An empty mapping if the file does not
		exist.
	:throws `RuntimeError` if `file` somehow couldn't be read from.
	"""
	try:
		with open(file, 'r') as handle:
			return json.load(handle)
	except FileNotFoundError:
		return {}
	except IOError as error:
		raise RuntimeError(f'An IO error occurred: \'{error}\'.')


def save_trl_questions_to_disk(trl_questions: TranslatedLCQuADQuestions,
                               file: Path) -> None:
	"""Saves the translated LC-QuAD 2.0 dataset split to `file`.

	:param trl_questions: A mapping from stringified UIDs to translated
		questions. Already-translated questions. May be incomplete or even
		empty.
	:param file: The file to which translated questions must be saved.
	:throws: `RuntimeError` when `file` could somehow not be written to.
	"""
	try:
		with open(file, 'w') as handle:
			json.dump(trl_questions, handle)
	except FileNotFoundError:
		raise RuntimeError(f'File \'{file.resolve()}\' was not found!')
	except IOError as error:
		raise RuntimeError(f'An IO error occurred: \'{error}\'.')


def question_uid_partition(ds_split: List[LCQuADQAPair],
                           trl_questions: TranslatedLCQuADQuestions,
						   save_freq: int) -> List[Set[int]]:
	"""Returns a partition of question UIDs.

	This function partitions the question UIDs present in `ds_split` but not
	in `trl_questions`: the 'remaining question UIDs', if you will.

	All or all-but-one parts of the partition contain `save_freq` question
	UIDs. A potential last part contains strictly less than `save_freq` question
	UIDs, but minimally one such UID.

	:param ds_split: The LC-QuAD 2.0 dataset split to derive a partition from.
	:param trl_questions: A mapping from stringified UIDs to translated
		questions. Already-translated questions. May be incomplete or even
		empty.
	:param save_freq: The number of questions to translate before performing
		a(nother) incremental save to disk. Must minimally be 1.
	"""
	assert save_freq >= 1
	uids = list(questions_still_to_translate(ds_split, trl_questions))
	start_index = 0  # inclusive
	partition: List[Set[int]] = []
	while start_index < len(uids):
		stop_index = min(start_index + save_freq, len(uids))  # exclusive
		partition.append(set(uids[start_index:stop_index]))
		start_index += save_freq
	return partition


def summarised_uids_set(uids_set: Set[int]) -> str:
	"""Returns a string that displays `uids_set` consisely.

	:param uids_set: The set of UIDs to summarise.
	:returns: The summary.
	"""
	if len(uids_set) <= 3:
		return '{%s}' % (', '.join(str(uid) for uid in uids_set),)
	else:
		first_two = ', '.join(str(uid) for uid in list(uids_set)[:2])
		last = list(uids_set)[-1]
		return '{%s, ..., %s}' % (first_two, last)


def translate_complete_dataset_split_questions(split: Split,
                                               language: NaturalLanguage,
                                               file: Path,
											   save_freq: int,
											   quiet: bool) -> None:
	"""Translates all questions of an LC-QuAD 2.0 dataset split into `language`
	and incrementally saves these to `file`.

	:param split: The LC-QuAD 2.0 dataset split to translate questions of.
	:param language: The language to translate questions into.
	:param file: The file to incrementally save translations into.
	:param save_freq: The number of questions to translate before performing
		a(nother) incremental save to `file`. Must minimally be 1.
	:param quiet: Whether to suppress indication of progress (`True`) or to
		show it (`False`).
	"""
	assert save_freq >= 1
	_ = load_dotenv()  # for loading the `.env`
	ds_split = dataset_split(split)
	trl_questions = trl_questions_from_disk(file)
	partition = question_uid_partition(ds_split, trl_questions, save_freq)
	if not quiet:	
		overwritably_print('Starting translation of split %s into \'%s\'...' %
		                   (split, language))
	for index, part in enumerate(partition):
		translate_dataset_split_questions(ds_split,
		                                  trl_questions,
										  language,
										  question_uids=part)
		save_trl_questions_to_disk(trl_questions, file)
		if not quiet:
			percent_done = ((index + 1) / len(partition)) * 100.
			uids_s = summarised_uids_set(uids_set=part)
			overwritably_print('Saved translated questions ' +
			                   'with UIDs %26s (%6.2lf/100%%).' %
			                   (uids_s, percent_done))
