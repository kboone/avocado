#!/usr/bin/env sh

# Augment the training set and extract features on it.
python ./code/augment_training.py

# Preprocess the data and run the Gaussian process fits on each object.
python ./code/featurize_all_test.py
