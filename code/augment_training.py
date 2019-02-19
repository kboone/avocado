from plasticc import Dataset

from settings import settings

# Load the training dataset
d_train = Dataset()
d_train.load_training_data()

# Build the augmented training set
num_augments = settings['NUM_AUGMENTS']
print("Augmenting training dataset with %d augments" % num_augments)
d_aug = d_train.augment_dataset(num_augments)

# Compute and save features on the augmented training set.
print("Extracting features on augmented dataset")
d_aug.extract_all_features()
