import json
import numpy as np
import os

from typing import Any, Dict, Tuple

from pre_processing.dataset.lc_quad import LCQuAD


ADDENDA_PATH = 'resources/datasets/lcquad/addenda.json'


if __name__ == '__main__':
	lcq = LCQuAD()
	previous_add: Dict[int, Any] = {}
	if os.path.exists(ADDENDA_PATH):
		with open(ADDENDA_PATH, 'r') as handle:
			previous_add = json.load(handle)
	index_range = (22270, 26770)
	ran: Tuple[int, ...] = tuple(np.arange(*index_range, 50))
	for i in range(len(ran) - 1):
		start: int = ran[i]
		end: int = ran[i + 1]
		print('RANGE (%4d, %4d)' % (start, end))
		add = lcq.question_addenda((start, end))
		with open(ADDENDA_PATH, 'w') as handle:
			previous_add.update(add)
			json.dump(previous_add, handle)
