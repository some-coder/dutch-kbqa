#!/usr/bin/env bash


# Ensure 'wget' exists.
if ! type "wget" > /dev/null; then
	echo "Couldn't find command 'wget'. Please install it first."
	exit
fi


# Download the LC-QuAD 2.0 training and testing datasets.
FIGSHARE_BASE_URL="https://figshare.com/ndownloader/files"
TRAIN_FILE_ID="15738824"
TEST_FILE_ID="15738818"
DATASET_DIRECTORY="${PWD}/resources/dataset"
TRAIN_FILE_NAME="train-${SOURCE_LANGUAGE}.json"
TEST_FILE_NAME="test-${SOURCE_LANGUAGE}.json"

if [ -d "$DATASET_DIRECTORY" ] ; then
	:
else
	mkdir -p "$DATASET_DIRECTORY"
	echo "Created directory \"$DATASET_DIRECTORY\"."
fi

echo -n "Downloading datasets from \"$FIGSHARE_BASE_URL\"... "
wget --no-check-certificate \
     --quiet \
     "$FIGSHARE_BASE_URL/$TRAIN_FILE_ID" \
     -O "$DATASET_DIRECTORY/$TRAIN_FILE_NAME"
wget --no-check-certificate \
     --quiet \
     "$FIGSHARE_BASE_URL/$TEST_FILE_ID" \
     -O "$DATASET_DIRECTORY/$TEST_FILE_NAME"
echo "Done."

