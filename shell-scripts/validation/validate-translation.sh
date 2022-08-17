#!/usr/bin/env bash


source source/py-ds-create/.venv/bin/activate

python3 source/py-ds-create/dutch_kbqa_py_ds_create/main.py \
	--task "validate-translation" \
	--save_file_name "${SPLIT}-${TARGET_LANGUAGE}.json" \
	--reference_file_name "$REFERENCE_TRANSLATED_FILE"

deactivate

