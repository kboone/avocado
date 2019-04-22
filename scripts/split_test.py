"""
Split the PLAsTiCC test dataset into small chunks that can be processed
individually.
"""

import numpy as np
import pandas as pd
import tqdm
import os

from settings import settings

test_meta_data = pd.read_csv(settings["RAW_TEST_METADATA_PATH"])
test_path = settings["RAW_TEST_PATH"]
split_path_file_format = settings["SPLIT_TEST_PATH_FORMAT"]

# Find the object id for the last line in the file.
print("Finding last object id")
with open(test_path, 'rb') as f:
    f.seek(-2, os.SEEK_END)
    while f.read(1) != b'\n':
        f.seek(-2, os.SEEK_CUR)
    last_line = f.readline()
    last_object_id = int(last_line.decode('ascii').split(',')[0])

remainder = None

chunk_size = 2*10**6

print("Splitting test set")
for chunk_idx, chunk in tqdm.tqdm(
        enumerate(pd.read_csv(test_path, chunksize=chunk_size))):
    if remainder is not None:
        chunk = pd.concat([remainder, chunk])

    # If we aren't on the last entry, chop off the end and keep it for the next
    # entry.
    chunk_last_object_id = chunk.iloc[-1]['object_id']
    if last_object_id != chunk_last_object_id:
        # Not the last one, need to chop off the last few rows and add them to
        # the remainder dataframe
        mask = chunk['object_id'] == chunk_last_object_id
        remainder = chunk[mask]
        chunk = chunk[~mask]

    object_ids = np.unique(chunk['object_id'])

    meta_chunk = test_meta_data[test_meta_data['object_id'].isin(object_ids)]

    # Write out the splits.
    meta_chunk.to_hdf(split_path_file_format % chunk_idx, key='meta', mode='w')
    chunk.to_hdf(split_path_file_format % chunk_idx, key='df')
