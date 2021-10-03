"""
Methods for translating questions from a KBQA dataset into another language.
"""


import csv
import os
import six

from google.cloud import translate_v2 as tl
from pathlib import Path

from typing import Dict, cast

from pre_processing.language import NaturalLanguage
from pre_processing.question import StringQuestion
from pre_processing.dataset.dataset import Dataset


ACCESS_CONFIRM_STRING = 'Truly access Google Cloud Translate? This may cost money! (y/n) '
KEY_FILE_LOCATION = Path('resources', 'google-cloud-key.json')
TRANSLATED_TEXT_KEY = 'translatedText'
TRANSLATIONS_DIRECTORY = Path('translations')


def _confirm_access() -> None:
	access_confirmation_response: str = input(ACCESS_CONFIRM_STRING)
	while access_confirmation_response not in ('y', 'n'):
		access_confirmation_response = input(ACCESS_CONFIRM_STRING)
	if access_confirmation_response == 'n':
		print('Not accessing Google Cloud Translate. Aborting...')
		exit(0)


def translate_text(text: StringQuestion, language: NaturalLanguage) -> str:
	client = tl.Client()
	if isinstance(text, six.binary_type):
		# convert the text to UTF-8
		text: six.binary_type
		text: str = text.decode('utf-8')
	result: Dict[str, str] = cast(Dict[str, str], client.translate(text, target_language=language.value))
	return result[TRANSLATED_TEXT_KEY]


def translate_for_dataset(dataset: Dataset, language: NaturalLanguage) -> None:
	"""
	Translates all questions present in ``dataset`` in the language ``language``.

	The output translations are saved to disk, to avoid unnecessary re-computations on the cloud's side.

	:param dataset: The dataset to translate for.
	:param language: The language to translate into.
	"""
	_confirm_access()
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path(os.getcwd(), KEY_FILE_LOCATION))
	translation_dir = Path(TRANSLATIONS_DIRECTORY, dataset.__class__.__name__.lower())
	qs = dataset.questions_for_translation()
	try:
		os.makedirs(translation_dir)
	except FileExistsError:
		pass  # that's okay
	with open(Path(translation_dir, 'translations.csv'), 'x') as handle:
		writer = csv.writer(handle)
		print('TRANSLATION PROGRESS')
		for index, q in enumerate(qs[:5]):
			translation = translate_text(q, language)
			writer.writerow([index, translation])
			print('\t%5d / %5d = %6.3f' % (index + 1, len(qs), ((index + 1) / len(qs)) * 1e2))
