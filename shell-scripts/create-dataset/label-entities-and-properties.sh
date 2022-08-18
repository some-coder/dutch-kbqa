#!/usr/bin/env bash

cd source/cpp-ds-create

cmake --build build
./build/main \
	--task "label-entities-and-properties" \
	--split "$SPLIT" \
	--language "$LABEL_LANGUAGE" \
	--part-size 15 \
	--quiet "false"

cd ../..

