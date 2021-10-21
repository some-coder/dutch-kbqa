import json
import numpy as np
import os

from utility.language import NaturalLanguage
from pre_processing.question import QuestionForm

from typing import Any, Dict, Tuple

from pre_processing.dataset.lc_quad import LCQuAD


ADDENDA_PATH = 'resources/datasets/lcquad/addenda.json'


def create_addenda():
	lcq = LCQuAD()
	previous_add: Dict[int, Any] = {}
	if os.path.exists(ADDENDA_PATH):
		with open(ADDENDA_PATH, 'r') as handle:
			previous_add = json.load(handle)
	index_range = (30220, 30225)
	ran: Tuple[int, ...] = tuple(np.arange(*index_range))
	for i in range(len(ran) - 1):
		start: int = ran[i]
		end: int = ran[i + 1]
		print('RANGE (%4d, %4d)' % (start, end))
		add = lcq.question_addenda((start, end))
		with open(ADDENDA_PATH, 'w') as handle:
			previous_add.update(add)
			json.dump(previous_add, handle)


if __name__ == '__main__':
	lc_q = LCQuAD()
	print(lc_q.qa_pairs[43].q.in_form(QuestionForm.BRACKETED_ENTITIES, NaturalLanguage.ENGLISH))
	print(lc_q.qa_pairs[43].q.in_form(QuestionForm.BRACKETED_ENTITIES_RELATIONS, NaturalLanguage.ENGLISH))
