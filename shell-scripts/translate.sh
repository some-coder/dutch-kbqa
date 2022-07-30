#!/usr/bin/env bash


source source/py-ds-create/.venv/bin/activate

python3 source/py-ds-create/dutch_kbqa_py_ds_create/main.py \
	--task "translate" \
	--split "train" \
	--language "nl" \
	--save_file_name "train_nl.json" \
	--save_frequency 10 \
	--quiet "false"

deactivate

