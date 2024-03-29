#!/usr/bin/env bash

cd source/cpp-ds-create

cmake --build build
./build/main \
	--task "mask-question-answer-pairs" \
	--split "$SPLIT" \
	--language "$TARGET_LANGUAGE" \
	--quiet "false"

cd ../..

