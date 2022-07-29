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
DATASET_DIRECTORY="resources"
TRAIN_FILE_NAME="train.json"
TEST_FILE_NAME="test.json"

mkdir "$DATASET_DIRECTORY"
wget --no-check-certificate \
     --quiet \
     "$FIGSHARE_BASE_URL/$TRAIN_FILE_ID" \
     -O "$DATASET_DIRECTORY/$TRAIN_FILE_NAME"
wget --no-check-certificate \
     --quiet \
     "$FIGSHARE_BASE_URL/$TEST_FILE_ID" \
     -O "$DATASET_DIRECTORY/$TEST_FILE_NAME"

