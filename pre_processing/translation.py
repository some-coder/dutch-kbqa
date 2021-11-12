"""
Methods for translating questions from a KBQA dataset into another language.
"""


import csv
import json
import os
import six

from google.cloud import translate_v2 as tl
from pathlib import Path

from typing import Dict, List, Optional, Tuple, cast

from pre_processing.dataset.dataset import Dataset

from utility.language import NaturalLanguage


GOOGLE_APPLICATION_CREDENTIALS = 'GOOGLE_APPLICATION_CREDENTIALS'
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


def _configure_credentials() -> None:
	os.environ[GOOGLE_APPLICATION_CREDENTIALS] = str(Path(os.getcwd(), KEY_FILE_LOCATION))


def translate_text(text: str, language: NaturalLanguage) -> str:
	client = tl.Client()
	if isinstance(text, six.binary_type):
		# convert the text to UTF-8
		text: six.binary_type
		text: str = text.decode('utf-8')
	result: Dict[str, str] = cast(Dict[str, str], client.translate(text, target_language=language.value))
	return result[TRANSLATED_TEXT_KEY]


def translate_texts(texts: List[Tuple[int, str]], language: NaturalLanguage, save_file: str) -> None:
	_confirm_access()
	_configure_credentials()
	translation_dir = Path(TRANSLATIONS_DIRECTORY)
	try:
		os.makedirs(translation_dir)
	except FileExistsError:
		pass
	previous: Optional[Dict[int, str]] = None
	if os.path.exists(translation_dir / save_file):
		with open(translation_dir / save_file, 'r') as handle:
			previous = json.load(handle)
	with open(translation_dir / save_file, 'w') as handle:
		d: Dict[int, str] = {}
		for identifier, text in texts:
			d[identifier] = translate_text(text, language)
			print(d[identifier])
		if previous is None:
			json.dump(d, handle)
		else:
			previous.update(d)
			json.dump(previous, handle)


def translate_for_dataset(
		dataset: Dataset,
		language: NaturalLanguage,
		start: int,
		stop: Optional[int] = None,
		append: bool = False) -> None:
	"""
	Translates all questions present in ``dataset`` in the language ``language``.

	The output translations are saved to disk, to avoid unnecessary re-computations on the cloud's side.

	:param dataset: The dataset to translate for.
	:param language: The language to translate into.
	:param start: The starting index. Inclusive.
	:param stop: Optional. The stopping index. Exclusive.
	:param append: Whether to append instead of exclusively create. Defaults to ``False``.
	"""
	_confirm_access()
	_configure_credentials()
	translation_dir = Path(TRANSLATIONS_DIRECTORY, dataset.__class__.__name__.lower())
	qs = dataset.questions_for_translation()
	try:
		os.makedirs(translation_dir)
	except FileExistsError:
		pass  # that's okay
	with open(Path(translation_dir, 'translations.csv'), 'a' if append else 'x') as handle:
		writer = csv.writer(handle)
		stop: int = len(qs) if stop is None else stop
		print('TRANSLATION PROGRESS')
		for index, q in enumerate(qs[start:stop]):
			index += start
			translation = translate_text(q, language)
			writer.writerow([index, translation])
			print('\t%5d / %5d = %6.3f' % (index + 1, (stop - start + 1), ((index + 1) / (stop - start + 1)) * 1e2))
