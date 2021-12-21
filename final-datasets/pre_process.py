import json
import re
from pathlib import Path
from typing import Tuple, List
from enum import Enum
import math
import os


class PreProcessingMode(Enum):
	"""
	An indicator symbol that communicates to the program what part of the data you'd like to process.
	"""
	TRAIN = 'train'  # includes validation ('dev')
	TEST = 'test'


def query_with_replaced_variables(query: str) -> str:
	"""
	Returns the SPARQL query with normalised variables.

	:param query: The query for which to replace the variables.
	:returns: The normalised SPARQL query.
	"""
	matches = tuple(set(m.group() for m in re.finditer('(\?)[^ ]+', query)))
	for index, m in enumerate(matches):
		query = re.sub('(' + re.sub('(\?)', '\\?', m) + ')', 'var_' + str(index + 1), query)
	return query


def preprocessed_question(raw_question: str) -> str:
	out = raw_question.lower()
	out = re.sub('[pq][0-9]+', lambda m: ' %s ' % (m.group(),), out)  # space around variables
	out = re.sub('(\?)$', ' ?', out)
	out = re.sub('[ ]{2,}', ' ', out)  # excess whitespace
	return out


def preprocessed_answer(raw_answer: str) -> str:
	out = raw_answer.lower()
	out = re.sub('[\{]', ' brack_open ', out)
	out = re.sub('[\}]', ' brack_close ', out)
	out = re.sub('[\(]', ' attr_open ', out)
	out = re.sub('[\)]', ' attr_close ', out)
	out = re.sub('[\.]', ' sep_dot ', out)
	out = re.sub('[,]', ' , ', out)
	out = re.sub('([a-z]+:)(?=[pq][0-9]+)', '', out)  # namespaces
	out = query_with_replaced_variables(out)
	out = re.sub('[ ]{2,}', ' ', out)  # excess whitespace
	out = re.sub('( )+$', '', out)  # trailing whitespace
	return out


def preprocessed_pair(raw_pair: Tuple[str, str]) -> Tuple[str, str]:
	"""
	Pre-processes a single QA pair.

	:param raw_pair: The yet-unprocessed QA pair.
	:returns: The processed QA pair.
	"""
	return preprocessed_question(raw_pair[0]), preprocessed_answer(raw_pair[1])


def raw_qa_pairs(location: Path) -> List[Tuple[str, str]]:
	"""
	Obtains all raw QA pairs stored in the file at the specified location.

	:param location: The file's location. Must be a JSON file.
	:returns: The raw QA pairs.
	"""
	with open(location, 'r') as handle:
		return json.load(handle)


if __name__ == '__main__':
	# pre-processing settings
	mode = PreProcessingMode.TRAIN
	percent_validate = 5e2 / (4e3 + 5e2)  # as used in LC-QuAD (around 11.1% to validation)
	out_dir = 'out'

	# perform actual pre-processing
	raw =  raw_qa_pairs('final_qa_pairs%s.json' % ('_test' if mode == PreProcessingMode.TEST else ''))
	inp = []
	out = []
	for key, value in raw.items():
		refined = preprocessed_pair((value['q'], value['a']))
		inp.append(refined[0] + '\n')
		out.append(refined[1] + '\n')
	
	# write to output directory
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	if mode == PreProcessingMode.TEST:
		with open(os.path.join(out_dir, 'test.nl'), 'w') as handle:
			handle.writelines(inp)
		with open(os.path.join(out_dir, 'test.sparql'), 'w') as handle:
			handle.writelines(out)
	else:
		number_validate = math.floor(len(inp) * percent_validate)
		with open(os.path.join(out_dir, 'train.nl'), 'w') as handle:
			handle.writelines(inp[number_validate:])
		with open(os.path.join(out_dir, 'train.sparql'), 'w') as handle:
			handle.writelines(out[number_validate:])
		with open(os.path.join(out_dir, 'dev.nl'), 'w') as handle:
			handle.writelines(inp[:number_validate])
		with open(os.path.join(out_dir, 'dev.sparql'), 'w') as handle:
			handle.writelines(out[:number_validate])