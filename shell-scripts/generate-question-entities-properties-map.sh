#!/usr/bin/env bash


cd source/cpp-ds-create

cmake --build build
./build/main \
	--task "generate-question-entities-properties-map" \
	--split "$SPLIT"

cd ../..

