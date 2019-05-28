***************
Reference / API
***************

.. currentmodule:: avocado


Datasets
========

*Loading/saving a dataset*

.. autosummary::
   :toctree: api

   Dataset
   Dataset.load
   Dataset.from_objects
   Dataset.path
   Dataset.write

*Retrieving objects from the dataset*

.. autosummary::
   :toctree: api

   Dataset.get_object

*Plotting lightcurves of objects in the dataset*

.. autosummary::
   :toctree: api

   Dataset.plot_light_curve
   Dataset.plot_interactive

*Extracting features from objects in the dataset*

.. autosummary::
   :toctree: api

   Dataset.extract_raw_features
   Dataset.get_raw_features_path
   Dataset.write_raw_features
   Dataset.load_raw_features
   Dataset.select_features

*Classifying objects in the dataset*

.. autosummary::
   :toctree: api

   Dataset.predict
   Dataset.get_predictions_path
   Dataset.write_predictions
   Dataset.load_predictions
   Dataset.label_folds


Astronomical objects
====================

.. autosummary::
   :toctree: api

   AstronomicalObject
   AstronomicalObject.bands
   AstronomicalObject.subtract_background
   AstronomicalObject.preprocess_observations
   AstronomicalObject.fit_gaussian_process
   AstronomicalObject.get_default_gaussian_process
   AstronomicalObject.predict_gaussian_process
   AstronomicalObject.plot_light_curve
   AstronomicalObject.print_metadata


Dataset augmentation
====================

*Augmentor API*

.. autosummary::
   :toctree: api

   Augmentor
   Augmentor.augment_object
   Augmentor.augment_dataset

*Augmentor Implementations*

.. autosummary::
   :toctree: api

   plasticc.PlasticcAugmentor

*Augmentor methods to implement in subclasses*

.. autosummary::
   :toctree: api

   Augmentor._augment_metadata
   Augmentor._choose_sampling_times
   Augmentor._choose_target_observation_count
   Augmentor._simulate_light_curve_uncertainties
   Augmentor._simulate_detection


Classification
==============

*Classifier API*

.. autosummary::
   :toctree: api

   Classifier
   Classifier.train
   Classifier.predict
   Classifier.path
   Classifier.write
   Classifier.load

*Classifier Implementations*

.. autosummary::
   :toctree: api

   LightGBMClassifier

*Weights and metrics*

.. autosummary::

   evaluate_weights_flat
   evaluate_weights_redshift
   weighted_multi_logloss


Feature extraction
==================

*Featurizer API*

.. autosummary::
   :toctree: api

   Featurizer
   Featurizer.extract_raw_features
   Featurizer.select_features
   Featurizer.extract_features

*Featurizer Implementations*

.. autosummary::
   :toctree: api

   plasticc.PlasticcFeaturizer

Exceptions
==========

.. autosummary::
   :toctree: api

   AvocadoException
