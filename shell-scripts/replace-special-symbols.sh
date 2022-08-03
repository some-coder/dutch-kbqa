#!/usr/bin/env bash

cd source/cpp-ds-create

cmake --build build
./build/main \
	--task "replace-special-symbols" \
	--load-file-name "${SPLIT}_nl" \
	--save-file-name "${SPLIT}_nl_replaced"

cd ../..

