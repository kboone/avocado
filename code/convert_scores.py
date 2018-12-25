import plasticc
import glob
import numpy as np
import pickle
import pandas as pd
import tqdm

print("Loading meta")
all_meta = []
chunk_d = plasticc.Dataset()
num_feature_files = len(glob.glob('../features/features_v2_test_*.h5'))
for chunk_id in tqdm.tqdm(range(num_feature_files)):
    chunk_d.load_chunk(chunk_id, load_flux_data=False)

    all_meta.append(chunk_d.meta_data)

meta = pd.concat(all_meta)

print("Loading scores")
scores = np.load('scores.npz')['scores']

print("Converting scores")
pred_df = plasticc.convert_scores(meta, scores)
pred_df.to_csv('./pred.csv')
