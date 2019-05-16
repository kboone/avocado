#!/usr/bin/env python
"""Script to extract the features for all chunks of the test set"""

import glob
import tqdm

from featurize_data import featurize_chunk
from settings import settings

num_chunks = len(glob.glob('%s/plasticc_split_*.h5' %
                           settings['SPLIT_TEST_DIR']))
print("Found %d chunks" % num_chunks)

print("Featurizing test set in chunks")
for chunk in tqdm.tqdm(range(num_chunks)):
    featurize_chunk(chunk)
