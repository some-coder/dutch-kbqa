#!/usr/bin/bash


python3 run.py \
        --do_train \
        --do_eval \
        --model_type bert \
        --model_architecture bert2bert \
        --encoder_model_name_or_path bert-base-multilingual-cased \
        --decoder_model_name_or_path bert-base-multilingual-cased \
        --source sparql \
        --target nl \
        --train_filename ../final-datasets/out/train \
        --dev_filename ../final-datasets/out/dev \
        --output_dir /data/s3366235/dutch-kbqa/encode-decode/saved-models \
        --max_source_length 64 \
        --weight_decay 0.01 \
        --max_target_length 128 \
        --beam_size 10 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --learning_rate 5e-5 \
        --save_inverval 10 \
        --num_train_epochs 150 

