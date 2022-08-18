#!/usr/bin/env bash

cd source/cpp-ds-create

cmake --build build
./build/main \
	--task "replace-special-symbols" \
	--load-file-name "${SPLIT}-${TARGET_LANGUAGE}" \
	--save-file-name "${SPLIT}-${TARGET_LANGUAGE}-replaced"

cd ../..

