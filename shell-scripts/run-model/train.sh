#!/usr/bin/env bash


source source/py-model/.venv/bin/activate

python3 source/py-model/dutch_kbqa_py_model/main.py \
	--model_type "$MODEL_TYPE" \
	--model_architecture "$MODEL_ARCHITECTURE" \
	--encoder_id_or_path "$ENCODER_ID_OR_PATH" \
	--decoder_id_or_path "$DECODER_ID_OR_PATH" \
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

deactivate

