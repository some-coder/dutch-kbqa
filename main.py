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
	first_qa = lc_q.qa_pairs[43]
	print('Question forms:')
	for qf in first_qa.q.question_forms:
		for nl in first_qa.q.natural_languages:
			print('\t(%s, %s) %s' % (qf.value, nl.value, first_qa.q.in_form(qf, nl)))
	print('\nAnswer forms:')
	for af in first_qa.a.answer_forms:
		for fl in first_qa.a.formal_languages:
			print('\t(%s, %s) %s' % (af.value, fl.value, first_qa.a.in_form(af, fl)))

