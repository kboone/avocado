import plasticc_gp
import glob
import numpy as np
import pickle
import pandas as pd
import tqdm

print("Loading meta")
all_meta = []
chunk_d = plasticc_gp.Dataset()
num_feature_files = len(glob.glob('../features/features_v1_test_*.h5'))
for chunk_id in tqdm.tqdm(range(num_feature_files)):
    chunk_d.load_chunk(chunk_id, load_flux_data=False)

    all_meta.append(chunk_d.meta_data)

meta = pd.concat(all_meta)

print("Loading scores")
# scores = pickle.load(open('scores_40_lr0.05_2500.pkl', 'rb'))
# scores = pickle.load(open('scores_40_feat2_fold_minwt2000.pkl', 'rb'))
# scores = np.load('scores_40_feat2_fold_leaves50.npz')['scores']
scores = np.load('scores_40_aug3.npz')['scores']

print("Converting scores")
pred_df = plasticc_gp.convert_scores(meta, scores)
pred_df.to_csv('./pred_40_aug3.csv')
