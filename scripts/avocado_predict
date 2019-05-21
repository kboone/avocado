#!/usr/bin/env python
"""Generate predictions for a dataset using avocado."""

import argparse
from tqdm import tqdm

import avocado


def process_chunk(classifier, chunk, args, verbose=True):
    # Load the dataset
    if verbose:
        print("Loading dataset...")
    dataset = avocado.load(args.dataset, metadata_only=True, chunk=chunk,
                           num_chunks=args.num_chunks)
    dataset.load_raw_features()

    # Generate predictions.
    if verbose:
        print("Generating predictions...")
    predictions = dataset.predict(classifier)

    # Write the predictions to disk.
    if verbose:
        print("Writing out predictions...")
    dataset.write_predictions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help='Name of the dataset to train on.'
    )
    parser.add_argument(
        'classifier',
        help='Name of the classifier to use.'
    )
    parser.add_argument(
        '--num_chunks',
        type=int,
        default=100,
        help='The dataset will be processed in chunks to avoid loading all of '
        'the data at once. This sets the total number of chunks to use. '
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--chunk',
        type=int,
        default=None,
        help='If set, only process this chunk of the dataset. This is '
        'intended to be used to split processing into multiple jobs.'
    )

    args = parser.parse_args()

    # Load the classifier
    classifier = avocado.load_classifier(args.classifier)

    if args.chunk is not None:
        # Process a single chunk
        process_chunk(classifier, args.chunk, args)
    else:
        # Process all chunks
        print("Processing the dataset in %d chunks..." % args.num_chunks)
        for chunk in tqdm(range(args.num_chunks), desc='Chunk',
                          dynamic_ncols=True):
            process_chunk(classifier, chunk, args, verbose=False)

    print("Done!")
