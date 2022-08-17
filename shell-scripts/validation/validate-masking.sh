#!/usr/bin/env bash


source source/py-ds-create/.venv/bin/activate

python3 source/py-ds-create/dutch_kbqa_py_ds_create/main.py \
	--task "validate-masking" \
	--save_file_name "${SPLIT}-${TARGET_LANGUAGE}-replaced-no-errors-masked.json" \
	--reference_file_name "$REFERENCE_MASKED_FILE"

deactivate

