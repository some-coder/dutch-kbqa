"""
Methods for translating Python-interpretable question-answer pairs to new languages, like Dutch.
"""


import csv
import os
import six

from constants import GOOGLE_CLOUD_KEY_FILE, ORIGINAL_DATA_FILE, TRANSLATIONS_FILE
from convert import EntityLocatingTechnique, Language, QAPair, qa_pairs_from_json
from google.cloud import translate_v2 as tl
from pathlib import Path
from typing import Dict, List, Tuple, cast


def translate_text(text: str, target_language: Language) -> str:
	"""
	Translates the supplied text to the given target language.

	:param text: The text to translate.
	:param target_language: The target language to translate to.
	:returns: The translated text.
	"""
	client = tl.Client()
	if isinstance(text, six.binary_type):
		text = text.decode('utf-8')
	result: Dict[str, str] = cast(Dict[str, str], client.translate(text, target_language=target_language.value))
	return result['translatedText']


def translate_questions(qas: List[QAPair], target_language: Language, working_range: Tuple[int, int]) -> Dict[int, str]:
	"""
	Translates the questions of the supplied QA pairs, working in the specified range.

	:param qas: The QA pairs of which to process the questions.
	:param target_language: The language to translate to.
	:param working_range: The index range from which to grab QA pairs. Inclusive, exclusive.
	:returns: A mapping from QA pair identifier codes to translated sentences.
	"""
	results: Dict[int, str] = {}
	if working_range[0] > working_range[1]:
		raise ValueError(
			'Start of working range must lie before the range\'s end, but %d > %d!' %
			(working_range[0], working_range[1]))
	start: int = max(0, working_range[0])
	end: int = min(len(qas), working_range[1])
	with open(TRANSLATIONS_FILE, 'a') as handle:
		writer = csv.writer(handle)
		for index in range(start, end):
			print(
				'(index: %4d) %4d/%4d (%5.2lf%%)' %
				(index + 1, start - index + 1, end - start, ((start - index + 1) / (end - start)) * 1e2))
			identifier: int = qas[index].identifier
			text: str = qas[index].q.form(Language.ENGLISH, EntityLocatingTechnique.WITH_BRACKETS, question_mark=True)
			translation: str = translate_text(text, target_language)
			writer.writerow([index, identifier, translation])
			if identifier in results:
				raise KeyError('Found a duplicate key (index=%d, ID=%d), aborting!' % (index, identifier))
			results[identifier] = translation
	return results


if __name__ == '__main__':
	# grab all translations from Google Cloud Translate and store it in a CSV file
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(), GOOGLE_CLOUD_KEY_FILE)
	questions_answers = qa_pairs_from_json(Path(ORIGINAL_DATA_FILE))
	translate_questions(questions_answers, Language.DUTCH, (0, len(questions_answers)))
