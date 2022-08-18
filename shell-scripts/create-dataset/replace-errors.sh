#!/usr/bin/env bash


source source/py-ds-create/.venv/bin/activate

python3 source/py-ds-create/dutch_kbqa_py_ds_create/main.py \
	--task "replace-errors" \
	--load_file_name "${SPLIT}-${TARGET_LANGUAGE}-replaced.json" \
	--save_file_name "${SPLIT}-${TARGET_LANGUAGE}-replaced-no-errors.json" \
	--split "$SPLIT" \
	--language "${TARGET_LANGUAGE}"

deactivate

