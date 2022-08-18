#!/usr/bin/env bash


source source/py-ds-create/.venv/bin/activate

python3 source/py-ds-create/dutch_kbqa_py_ds_create/main.py \
	--task "finalise-dataset" \
	--split "$SPLIT" \
	--language "${TARGET_LANGUAGE}" \
	--fraction_to_validate $VALIDATION_FRACTION

deactivate

