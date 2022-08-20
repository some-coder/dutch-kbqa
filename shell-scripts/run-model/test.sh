#!/usr/bin/env bash


source source/py-model/.venv/bin/activate

python3 source/py-model/dutch_kbqa_py_model/main.py \
	--model_type "$MODEL_TYPE" \
	--model_architecture "$MODEL_ARCHITECTURE" \
	--encoder_id_or_path "$ENCODER_ID_OR_PATH" \
	--decoder_id_or_path "$DECODER_ID_OR_PATH" \
	--natural_language "$NATURAL_LANGUAGE" \
	--query_language "$QUERY_LANGUAGE" \
	--max_natural_language_length $MAX_NATURAL_LANGUAGE_LENGTH \
	--max_query_language_length $MAX_QUERY_LANGUAGE_LENGTH \
	--beam_size $BEAM_SIZE \
	--perform_training "false" \
	--perform_validation "false" \
	--perform_testing "true" \
	--save_dir "${MODELS_DIR}/${MODEL_NAME}-test" \
	--seed $SEED \
	--non_training_batch_size $NON_TRAINING_BATCH_SIZE \
	--load_model "${MODELS_DIR}/${MODEL_NAME}"
# Note that you can add more arguments if you wish to change the script; see
# the `.sample-env` for options.

deactivate

