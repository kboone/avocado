import plasticc
import glob
import numpy as np
import pickle
import pandas as pd
import tqdm

from settings import settings

print("Loading meta")
all_meta = []
chunk_d = plasticc.Dataset()
num_chunks = len(glob.glob('%s/plasticc_split_*.h5' %
                           settings['SPLIT_TEST_DIR']))
for chunk_id in tqdm.tqdm(range(num_chunks)):
    chunk_d.load_chunk(chunk_id, load_flux_data=False)

    all_meta.append(chunk_d.meta_data)

meta = pd.concat(all_meta)

print("Loading scores")
num_augments = settings['NUM_AUGMENTS']
scores = np.load(settings['SCORES_PATH_FORMAT'] % num_augments)['scores']

print("Converting scores")
pred_df = plasticc.convert_scores(meta, scores)
pred_df.to_csv(settings['SUBMISSIONS_PATH_FORMAT'] % num_augments)
