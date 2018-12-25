from plasticc import Dataset

# Load the training dataset
d_train = Dataset()
d_train = Dataset.load_training_data()

# Build the augmented training set
d_aug = d_train.augment_dataset(40)

# Compute and save features on the augmented training set.
d_aug.extract_all_features()
