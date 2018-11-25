import plasticc_gp
import glob
import pandas as pd
import tqdm

print("------ TRAINING ------")
d = plasticc_gp.Dataset()
# d.load_training_data()
# d.load_features()
d.load_augment(20)
# d.load_augment(10)
d.load_features()

classifiers = d.train_classifiers(do_fold=False)


print("------ PREDICTING ------")

all_pred = []
chunk_d = plasticc_gp.Dataset()
num_feature_files = len(glob.glob('../features/features_v1_test_*.h5'))
for chunk_id in tqdm.tqdm(range(num_feature_files)):
    chunk_d.load_chunk(chunk_id, load_flux_data=False)
    chunk_d.load_features()

    pred = plasticc_gp.do_predictions(chunk_d.meta_data['object_id'],
                                      chunk_d.features, classifiers)
    all_pred.append(pred)

pred_df = pd.concat(all_pred)
pred_df.to_csv('./pred_aug_single_20_8.csv')
