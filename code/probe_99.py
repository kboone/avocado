import pandas as pd
import numpy as np

dataset = pd.read_csv('./pred_40_aug3_combo.csv').set_index('object_id')

gal_classes = [6, 16, 53, 65, 92]
extgal_classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]

gal_vals = dataset[['class_%d' % i for i in gal_classes]]
extgal_vals = dataset[['class_%d' % i for i in extgal_classes]]

gal_sum = np.sum(gal_vals, axis=1)
extgal_sum = np.sum(extgal_vals, axis=1)

is_gal = gal_sum > extgal_sum

# Remove class 99 predictions.
orig_99_preds = dataset['class_99'].copy()
dataset['class_99'] = 0.
norm = np.sum(dataset, axis=1)
dataset = dataset.div(norm, axis=0)

# Galactic predictions -> keep original predictions for now.
dataset['class_99'][is_gal] = (orig_99_preds / norm)[is_gal]

pred_99_extgal = 0
c_vals = {
    '42': (1, 0.5),
    '62': (1, 0.3),
    '52': (1, 0.1),
    '95': (1, 0.1),
}
# c_vals = {
    # '15': (2, 0.024),
    # '42': (1, 0.316),
    # '52': (1, 0.118),
    # '62': (1, 0.297),
    # '64': (2, 0.009),
    # '67': (1, 0.044),
    # '88': (1, 0.005),
    # '90': (1, 0.032),
    # '95': (1, 0.100),
# }
extgal_c_total = np.sum([i[1] for i in c_vals.values()])
print(extgal_c_total)
for label, (weight, c_val) in c_vals.items():
    frac_99 = c_val * 2 / weight / extgal_c_total
    pred_99_extgal += frac_99 * dataset['class_%s' % label]
    # dataset['class_%s' % label] *= (1 - frac_99)

dataset['class_99'][~is_gal] = pred_99_extgal[~is_gal]

# dataset['class_99'][~is_gal] = 2 * dataset['class_95'][~is_gal]
# dataset['class_6'][~is_gal] = dataset['class_52'][~is_gal]
# dataset['class_52'][~is_gal] = 0.
# print(np.sum(dataset, axis=1))

# Do class 99 prediction
# test_class = 42
# dataset['class_99'][~is_gal] = 2.0 * dataset['class_%d' % test_class][~is_gal]

# Normalize
dataset = dataset.div(np.sum(dataset, axis=1), axis=0)

# dataset.to_csv('./probe_99_%d_2.0.csv' % test_class)
# dataset.to_csv('./probe_99_round5.csv')
# dataset.to_csv('./probe_99_full_95.csv')
dataset.to_csv('./probe_99_quad.csv')
