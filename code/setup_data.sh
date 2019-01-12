#!/usr/bin/env sh

cd data

# Download the data from kaggle and unzip it
kaggle competitions download -c PLAsTiCC-2018
unzip test_set.csv.zip
unzip test_set_metadata.csv.zip
unzip training_set.csv.zip

cd ..

# Fix permissions
chmod ug+r data/*

# Split the test dataset into smaller chunks
python code/split_test.py
