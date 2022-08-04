#!/usr/bin/env bash


source source/py-ds-create/.venv/bin/activate

python3 source/py-ds-create/dutch_kbqa_py_ds_create/main.py \
	--task "replace-errors" \
	--load_file_name "${SPLIT}_nl_replaced.json" \
	--save_file_name "${SPLIT}_nl_replaced_no_errors.json" \
	--split "$SPLIT" \
	--language "nl"

deactivate

