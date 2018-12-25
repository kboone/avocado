import pandas as pd
import numpy as np

dataset = pd.read_csv('./pred_40_aug3.csv').set_index('object_id')

gal_classes = [6, 16, 53, 65, 92]
extgal_classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]

gal_vals = dataset[['class_%d' % i for i in gal_classes]]
extgal_vals = dataset[['class_%d' % i for i in extgal_classes]]

gal_sum = np.sum(gal_vals, axis=1)
extgal_sum = np.sum(extgal_vals, axis=1)

is_gal = gal_sum > extgal_sum

test_class = 15

# Multi-class probability
# prob_99 = dataset['class_%d' % test_class].copy()

# Top class probability
prob_99 = (
    dataset['class_%d' % test_class] == np.max(dataset, axis=1)
).astype(float)

print(np.sum(prob_99))

for gal_class in gal_classes:
    dataset['class_%d' % gal_class] = 0.
for extgal_class in extgal_classes:
    dataset['class_%d' % extgal_class] = 0.

# Put the leftover predictions in the wrong bins.
dataset['class_6'][~is_gal] = (1 - prob_99)[~is_gal]
dataset['class_15'][is_gal] = (1 - prob_99)[is_gal]

# Do class 99 prediction
dataset['class_99'] = prob_99

# Normalize
# dataset = dataset.div(np.sum(dataset, axis=1), axis=0)
dataset.to_csv('./probe_99_%d_individual_max.csv' % test_class)
