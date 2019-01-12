import plasticc
import glob
import numpy as np
import pickle
import pandas as pd
import tqdm

from settings import settings

# Load the classifiers
num_augments = settings['NUM_AUGMENTS']
classifiers = plasticc.load_classifiers(settings['MODEL_PATH_FORMAT'] %
                                        num_augments)

# Do the predictions.
all_scores = []
chunk_d = plasticc.Dataset()
num_chunks = len(glob.glob('%s/plasticc_split_*.h5' %
                           settings['SPLIT_TEST_DIR']))
print("Doing predictions for %d chunks" % num_chunks)
for chunk_id in tqdm.tqdm(range(num_chunks)):
    chunk_d.load_chunk(chunk_id, load_flux_data=False)
    chunk_d.load_features()

    scores = plasticc.do_scores(chunk_d.meta_data['object_id'],
                                chunk_d.features, classifiers)
    all_scores.append(scores)

all_scores = np.hstack(all_scores)

# Save the results. With only 16 GB of RAM, saving to a nice format like HDF
# crashes here. For whatever reason, np.savez uses a lot less memory so we use
# that.
np.savez(settings['SCORES_PATH_FORMAT'] % num_augments, scores=all_scores)
