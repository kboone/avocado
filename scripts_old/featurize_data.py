#!/usr/bin/env python
"""Script to extract the features for a single chunk of the test set.

This script can be called on the command line with a single integer argument
that is the index of the chunk to process.
"""

import argparse

import plasticc


def featurize_chunk(chunk):
    dataset = plasticc.Dataset()
    dataset.load_chunk(chunk)

    dataset.extract_all_features()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('chunk', type=int)
    args = parser.parse_args()
    featurize_chunk(args.chunk)
