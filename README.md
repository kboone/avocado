# Kyle Boone's solution to the 2018 Kaggle PLAsTiCC challenge.

This repository contains the code to reproduce the winning solution to the
[2018 Kaggle PLAsTiCC challenge](https://www.kaggle.com/c/PLAsTiCC-2018). For
details on my approach, see the [overview of the
solution](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033). If you have
any questions or issues, please contact me at kyboone@gmail.com. An older
README with more descriptions of the various scripts and functions but with a
more complicated workflow that is incompatible with the Kaggle requirements can
be found at README\_OLD.md.

# ARCHIVE CONTENTS
- augment             : Folder for augmented datasets, created from the training data.
- code                : Directory containing all of the code used to generate the submissions.
- data                : Location where the raw data from Kaggle is stored.
- data_split          : Location where the test data is stored after being split into smaller chunks
- features            : Folder for features extracted from the various sets.
- model               : Directory containing serialized models.
- scores              : Location where the scores for the test set are output.
- submissions         : Output directory for the final submissions to Kaggle.
- SETTINGS.json       : Settings file to specify the various parameters of the model and directories to work in.

# HARDWARE: (The following specs were used to create the original solution)
Most of the code was run on a machine with the following specifications:

CentOS 6.5
Intel(R) Xeon(R) CPU E3-1270
16 GB memory

The Gaussian process fits were distributed across the previous machine and the
following ones, each running CentOS 6.5, to speed up the computation time to
under a day of wall time instead of ~4 days:
- Intel(R) Xeon(R) CPU E5-2620 with 64 GB RAM
- Intel(R) Xeon(R) CPU E5-2620L with 32 GB RAM

# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.7.2

# DATA SETUP

The following script does all of the data downloading and splits
the test set into smaller files that can actually be loaded into memory. This
assumes that the [Kaggle API](https://github.com/Kaggle/kaggle-api) is
installed. This will populate the data and data\_split directories, and will
overwrite any files that were previously there (although there shouldn't be any
need to ever rerun this).

    sh ./code/setup_data.sh    

# DATA PROCESSING

The data processing script augments the training set, does Gaussian process
fits and feature extraction for the augmented set, and then does all of the
feature extraction for the test set. This will overwrite the old augmented
set and features if they were calculated with the same parameters.
Incrementing FEATURES\_VERSION or AUGMENT\_VERSION in the SETTINGS.json file
will avoid overwriting the old files. Setting NUM\_AUGMENTS in the
SETTINGS.json file to choose how many augments of the training set to do (40 by
default). To run this script, from the main directory run:

    sh ./code/preprocess.sh

This will take a long time to run (~100 hours). As an alternative, the Gaussian
process fits can be performed in parallel on different machines using the
following embarassingly parallel script:

    python ./code/featurize_data.py [chunk id]

# MODEL TRAINING

The following script trains the model. The model weights will be saved to the
model directory, and any previous weights for the same number of augments will
be overwritten.

    python ./code/train_classifiers.py

# PREDICTION

The following script produces the final predictions and packages them for
submission. Two different files are output into the submissions folder: one
with a proper class 99 prediction, and one with a class 99 prediction that came
from probing the leaderboard (and is therefore not useful for actual science
cases).

    sh code/generate_submission.sh

# RUNNING ON A NEW TEST SET

To run on a new test set, replace the test\_set.csv and test\_set\_metadata.csv
files in the data directory and rerun the steps described above (skipping
training if desired). Note that all of the scripts described previously are
very modular and typically call several subscripts. They can easily be
recombined for different workflows.
