"""
Contains a method to match a string in some text, allowing for inaccuracies.
"""


import difflib as dl
import re

from typing import Optional, Tuple


def match(string: str, text: str, threshold: float) -> Optional[Tuple[int, int]]:
	"""
	Tries to find a word subsequence in ``text`` that is similar to ``string`` as determined by ``threshold``.

	:param string: The string to search for.
	:param text: The text within which to search for the string.
	:param threshold: A real number between zero and one. Similarity scores are measured against this threshold.
	:return: The start and end indices of the word subsequence, if it could be found.
	"""
	num_words: int = len(string.split(' '))
	text = re.sub('[-_]', ' ', text)
	split_text = text.split(' ')
	for i in range(len(split_text) - num_words + 1):
		piece = ' '.join(split_text[i:(i + num_words)])
		ratio = dl.SequenceMatcher(None, string, piece).ratio()
		if ratio >= threshold:
			start = len(' '.join(split_text[:i])) + (1 if i != 0 else 0)  # take into account extra space
			stop = start + len(piece)
			return start, stop
	return None  # Couldn't find a close-enough match.
