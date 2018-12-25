import plasticc_gp
import glob
import numpy as np
import pickle
import pandas as pd
import tqdm

print("------ TRAINING ------")
d = plasticc_gp.Dataset()
d.load_augment(40)
d.load_features()

classifiers = d.train_classifiers(do_fold=True)


print("------ PREDICTING ------")

all_scores = []
chunk_d = plasticc_gp.Dataset()
num_feature_files = len(glob.glob('../features/features_v1_test_*.h5'))
for chunk_id in tqdm.tqdm(range(num_feature_files)):
    chunk_d.load_chunk(chunk_id, load_flux_data=False)
    chunk_d.load_features()

    scores = plasticc_gp.do_scores(chunk_d.meta_data['object_id'],
                                   chunk_d.features, classifiers)
    all_scores.append(scores)

all_scores = np.hstack(all_scores)

# Save the results.
# np.savez('scores_40_aug1.npz', scores=all_scores)
np.savez('scores_40_aug3.npz', scores=all_scores)
# pickle.dump(all_scores, open('scores_40_aug2_lr0.03.pkl', 'wb'))
