#!/usr/bin/env bash


if [ "${PY_MODEL_PACKAGE_MANAGER}" = "conda" ]; then
	source "$(conda info --base)/etc/profile.d/conda.sh"
	conda activate source/py-model/.conda-env || {
		echo '`conda activate` failed. Is Conda initialised in this shell?';
		exit 1;
	}
elif [ "${PY_MODEL_PACKAGE_MANAGER}" = "pip" ]; then
	source source/py-model/.venv/bin/activate || {
		echo '`pip activate` failed. Is there a `.venv` in `source/py-model/`?';
		exit 1;
	}
else
	echo "Invalid `source/py-model` package manager: '${PY_MODEL_PACKAGE_MANAGER}'.";
	exit 1;
fi

export PYTHONHASHSEED=$SEED  # Should be set statically. See Stack Overflow question ID 25684349.
export CUBLAS_WORKSPACE_CONFIG=$MY_CUBLAS_WORKSPACE_CONFIG
python3 source/py-model/dutch_kbqa_py_model/main.py \
	--enc_model_type "$ENC_MODEL_TYPE" \
	--dec_model_type "$DEC_MODEL_TYPE" \
	--enc_id_or_path "$ENC_ID_OR_PATH" \
	--dec_id_or_path "$DEC_ID_OR_PATH" \
	--dataset_dir "$DATASET_DIR" \
	--natural_language "$NATURAL_LANGUAGE" \
	--query_language "$QUERY_LANGUAGE" \
	--max_natural_language_length $MAX_NATURAL_LANGUAGE_LENGTH \
	--max_query_language_length $MAX_QUERY_LANGUAGE_LENGTH \
	--learning_rate $LEARNING_RATE \
	--beam_size $BEAM_SIZE \
	--perform_training "true" \
	--perform_validation "true" \
	--perform_testing "false" \
	--save_dir "${MODELS_DIR}/${MODEL_NAME}" \
	--seed $SEED \
	--training_batch_size $TRAINING_BATCH_SIZE \
	--non_training_batch_size $NON_TRAINING_BATCH_SIZE \
	--weight_decay $WEIGHT_DECAY \
	--training_epochs $TRAINING_EPOCHS \
	--save_frequency $SAVE_FREQUENCY
# Note that you can add more arguments if you wish to change the script; see
# the `.sample-env` for options.

if [ "${PY_MODEL_PACKAGE_MANAGER}" = "conda" ]; then
	conda deactivate
elif [ "${PY_MODEL_PACKAGE_MANAGER}" = "pip" ]; then
	deactivate
fi

