"""
Splits datasets up unto triples of training, validation, and testing (sub)sets.
"""


import json

from constants import ORIGINAL_DATA_FILE, ORIGINAL_TRANSLATIONS_FILE, MODIFIED_TRANSLATIONS_FILE, SPLITS_LOCATION
from convert import QAPair
from integrate import qa_pairs_with_dutch

from enum import Enum
from pathlib import Path
from typing import cast, Dict, List, Tuple


class SplitStrategy(Enum):
	"""
	A splitting strategy to use. 'MCD' stands for 'Maximal Compositional Divergence'.
	"""
	MCD_1 = 'mcd-1'
	MCD_2 = 'mcd-2'
	MCD_3 = 'mcd-3'
	RANDOM = 'random'


def _split_specification_from_json(strategy: SplitStrategy) -> Tuple[List[int], List[int], List[int]]:
	"""
	Internal use. Infers which dataset entries should go in which subset, based on their indices.

	:param strategy: The strategy to split with.
	:returns: A triple of indices.
	:throws FileNotFoundError: If the split specification JSON file cannot be found.
	"""
	with open(Path(SPLITS_LOCATION, '%s.json' % (strategy.value,)), 'r') as handle:
		dat: Dict[str, List[int]] = json.load(handle)
	return dat['trainIdxs'], dat['devIdxs'], dat['testIdxs']


def split(qa_pairs: List[QAPair], strategy: SplitStrategy) -> Tuple[List[QAPair], List[QAPair], List[QAPair]]:
	"""
	Splits the provided list of QA pairs into training, validation, and test lists.

	:param qa_pairs: The QA pairs to split.
	:param strategy: The strategy to split with.
	:returns: A triple of QA pair lists, corresponding to training, validation, and test lists.
	"""
	spec = _split_specification_from_json(strategy)
	return cast(Tuple[List[QAPair], List[QAPair], List[QAPair]], tuple([qa_pairs[i] for i in g] for g in spec))


if __name__ == '__main__':
	with_dutch = qa_pairs_with_dutch(
		Path(ORIGINAL_DATA_FILE),
		Path(ORIGINAL_TRANSLATIONS_FILE),
		Path(MODIFIED_TRANSLATIONS_FILE))
	s = split(with_dutch, SplitStrategy.MCD_1)
