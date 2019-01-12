import plasticc
import glob
import numpy as np
import pickle
import pandas as pd
import tqdm

from settings import settings

print("------ TRAINING ------")
d = plasticc.Dataset()
d.load_augment(settings['NUM_CHUNKS'])
d.load_features()

classifiers = d.train_classifiers(do_fold=True)


print("------ PREDICTING ------")

all_scores = []
chunk_d = plasticc.Dataset()
num_chunks = len(glob.glob('%s/plasticc_split_*.h5' %
                           settings['SPLIT_TEST_PATH']))
print("Found %d chunks" % num_chunks)
for chunk_id in tqdm.tqdm(range(num_chunks)):
    chunk_d.load_chunk(chunk_id, load_flux_data=False)
    chunk_d.load_features()

    scores = plasticc.do_scores(chunk_d.meta_data['object_id'],
                                chunk_d.features, classifiers)
    all_scores.append(scores)

all_scores = np.hstack(all_scores)

# Save the results.
np.savez('scores.npz', scores=all_scores)
