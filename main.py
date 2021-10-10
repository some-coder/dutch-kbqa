import json
import os

from typing import Any, Dict

from pre_processing.dataset.lc_quad import LCQuAD


ADDENDA_PATH = 'resources/datasets/lcquad/addenda.json'


if __name__ == '__main__':
	lcq = LCQuAD()
	add = lcq.question_addenda((15, 160))
	print(add)
	previous_add: Dict[int, Any] = {}
	if os.path.exists(ADDENDA_PATH):
		with open(ADDENDA_PATH, 'r') as handle:
			previous_add = json.load(handle)
	with open(ADDENDA_PATH, 'w') as handle:
		add.update(previous_add)
		json.dump(add, handle)

