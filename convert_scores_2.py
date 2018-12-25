import plasticc_gp
import glob
import numpy as np
import pickle
import pandas as pd
import tqdm

print("Loading meta")
all_meta = []
all_s2n = []
chunk_d = plasticc_gp.Dataset()
num_feature_files = len(glob.glob('../features/features_v1_test_*.h5'))
for chunk_id in tqdm.tqdm(range(num_feature_files)):
    chunk_d.load_chunk(chunk_id, load_flux_data=False)
    chunk_d.load_features()

    # Calculate total signal-to-noise
    total_s2n = np.sqrt(
        chunk_d.features['total_s2n_0']**2 +
        chunk_d.features['total_s2n_1']**2 +
        chunk_d.features['total_s2n_2']**2 +
        chunk_d.features['total_s2n_3']**2 +
        chunk_d.features['total_s2n_4']**2 +
        chunk_d.features['total_s2n_5']**2
    )

    all_meta.append(chunk_d.meta_data)
    all_s2n.extend(total_s2n)

all_s2n = np.array(all_s2n)
meta = pd.concat(all_meta)

print("Loading scores")
# scores = pickle.load(open('scores_40_lr0.05_2500.pkl', 'rb'))
# scores = pickle.load(open('scores_40_feat2_fold_minwt2000.pkl', 'rb'))
# scores = np.load('scores_40_feat2_fold_leaves50.npz')['scores']
# scores = np.load('scores_40_aug2.npz')['scores']
scores = np.load('scores_40_aug2_lr0.03_nophotoz.npz')['scores']

print("Converting scores")
pred_df = plasticc_gp.convert_scores_2(meta, scores, all_s2n)
pred_df.to_csv('./pred_40_aug2_s2nscore_nophotoz.csv')
