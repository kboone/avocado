# Kyle Boone's code for the 2018 Kaggle PLAsTiCC challenge.

This repository contains the code for the winning solution to the [2018 Kaggle
PLAsTiCC challenge](https://www.kaggle.com/c/PLAsTiCC-2018). For details on my
approach, see the [overview of the
solution](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033).

This code is made up of a sequence of scripts that produce the final output one
step at a time. The main package is the `plasticc.py` file which contains the
Dataset class which is used to read the lightcurve and meta data, compute and
reload features and produce the final predictions.

## Installation

This package has the following dependencies:
- numpy
- pandas
- tqdm
- astropy
- george
- lightgbm
- matplotlib
- scipy
- scikit-learn

Set up a python environment with all of those
([anaconda](https://www.anaconda.com/) is recommended) and clone this
repository to get started. Next, download the Kaggle PLAsTiCC dataset and unzip
it into the data folder of this repository.

## Usage

This code is designed to be used interactively in a Jupyter notebook to explore
the dataset and test out different ideas. This section describes the
available methods. The following sections will describe several scripts that
link the methods together to build a full pipeline.

To load the training set, launch a Jupyter notebook from the `code` directory
and run the following methods:

    run plasticc.py
    
    d_train = Dataset()
    d_train.load_training_data()

The `Dataset` class has a lot of functionality. The following methods load a
different subset of the data:
- `load_training_data`: load the training dataset.
- `load_test_data`: load the full test dataset (meta data only).
- `load_chunk`: load a chunk of the test dataset (meta + flux data).
- `load_augment`: load an augmented version of the training set.

After a dataset is loaded, the next step is to fit a Gaussian process to the
data and compute features. This is done with the following methods:
- `fit_gp`: do the Gaussian process fit and return a dictionary with the fit
  results and predictions.
- `plot_gp`: plot the Gaussian process prediction.
- `plot_gp_interactive`: Show an interactive plot of Gaussian process fits to
  the full sample in a Jupyter notebook. This uses ipywidgets to build an
interactive display.
- `extract_features`: Extract features from a lightcurve using the results of
  the Gaussian process prediction.
- `extract_all_features`: Extract features for every lightcurve in the dataset,
  and then save them to an HDF file in the `features` directory.
- `load_features`: Load the features for a dataset that were previously
  extracted and saved.

Once the features have been computed/loaded, a LightGBM classifier can be
trained on the training dataset. This is done using `Dataset.train_classifiers`
which returns a list of classifiers trained on several folds. These classifiers
can then be used to score the test set using `do_scores`. Finally, the scores
are converted to probabilities for the different classes using the
`convert_scores` function.

An additional feature of this program is the ability to "augment" the dataset
by degrading the training data. To do this, load the original training data and
run `d_train.augment_dataset(40)`. This will produce an augmented dataset with
40 versions of each of the original lightcurves.

Finally, a postprocessing step is available to redo the class 99 predictions
using scores derived from leaderboard probing. This is done by running the
`apply_probe_99.py` script on the original output.

## Pipeline to produce the final results.

### Setup

First, download the PLAsTiCC dataset and unzip it into the `data` directory.
The test set is too large to load into memory for processing on my machine
(with 16 GB of RAM), so the pipeline works on it in chunks. Run the
`split_test.py` script to generate a set of 227 HDF files in the `data_split`
directory. These files will contain a `meta` key with the meta data for that
chunk and a `df` key with the flux dat for that chunk.

### Augment the training set

Run the `augment_training.py` script. This will load the training dataset,
augment it using up to 40 copies of each of the original lightcurves and then
extract features. This will take ~10 hours to run for 40 copies.

### Extract features on the test set.

The `featurize_data.py` script fits Gaussian processes and computes features on
the test set. This script takes an integer argument which is the chunk number
to use. This script can be run in parallel, or distributed across multiple
machines (recommended). On my machine, I can compute features for 10 objects
per second which means that it takes ~4 days of computation time to extract
features for every object. The features are saved to disk after this step is
run.

### Train the classifier and score the test set.

Run the `do_scores.py` script to train classifiers and compute scores on the
test set. By default, this will use a 40x augmented training set with 5-fold
cross-validation. This will output a `scores.npz` file with the scores for each
of the 5 classifiers. The cross-validated score on the original training set
will also be printed out.

### Convert the scores to the final prediction.

Run the `convert_scores.py` script to convert the scores to the final
prediction. This uses a "real" prediction for the outliers which is done by
assigning a flat score to the class 99 objects that goes into a softmax
probability prediction. The class 99 score is capped to be no larger than the
average of the two largest scores to avoid very high class 99 predictions for
noisy data. This will output a `pred.csv` file with predictions that can be
submitted to Kaggle.

### Apply a class 99 prediction that uses leaderboard probing for optimization.

As described in the solution overview, the class 99 prediction can be optimized
by probing the leaderboard. The `apply_probe_99.py` script updates the
extragalactic class 99 prediction using a combination of the other classes.
This outputs a `pred_probe.csv` file which is the final submission used in this
competition.
