#!/usr/bin/env bash


source source/py-ds-create/.venv/bin/activate

python3 source/py-ds-create/dutch_kbqa_py_ds_create/main.py \
	--task "translate" \
	--split "$SPLIT" \
	--language "$TARGET_LANGUAGE" \
	--save_file_name "${SPLIT}-${TARGET_LANGUAGE}.json" \
	--save_frequency $TRANSLATE_SAVE_FREQ \
	--quiet "false"

deactivate

