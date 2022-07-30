#!/usr/bin/env bash


source source/py-ds-create/.venv/bin/activate

python3 source/py-ds-create/dutch_kbqa_py_ds_create/main.py \
	--task "validate" \
	--save_file_name "train_nl.json" \
	--reference_file_name "/Volumes/SEAGATE EDM/contents/university/honours/master/master-work/project/cpp-pre-processing/data/data_nl.json"

deactivate

