import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from .settings import settings
from .utils import logger, AvocadoException


def get_classifier_path(name):
    """Get the path to where a classifier should be stored on disk

    Parameters
    ----------
    name : str
        The unique name for the classifier.
    """
    classifier_directory = settings["classifier_directory"]
    classifier_path = os.path.join(classifier_directory, "classifier_%s.pkl" % name)

    return classifier_path


def evaluate_weights_flat(dataset, class_weights=None):
    """Evaluate the weights to use for classification on a dataset.

    The weights are set to normalize each class to have same weight with the
    same weight for each object in a class. If class weights are set, those
    weights are applied after normalization.

    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to evaluate weights on.
    class_weights : dict (optional)
        Weights to use for each class. If not set, equal weights are assumed
        for each class.

    Returns
    -------
    weights : `pandas.Series`
        The weights that should be used for classification.
    """
    use_metadata = dataset.metadata

    object_classes = use_metadata["class"]
    class_counts = object_classes.value_counts()

    norm_class_weights = {}
    for class_name, class_count in class_counts.items():
        if class_weights is not None:
            class_weight = class_weights[class_name]
        else:
            class_weight = 1

        norm_class_weights[class_name] = (
            class_weight * len(object_classes) / class_count
        )

    weights = object_classes.map(norm_class_weights)

    return weights


def evaluate_weights_redshift(
    dataset,
    class_weights=None,
    group_key=None,
    min_redshift=None,
    max_redshift=None,
    num_bins=None,
    min_bin_count=None,
    redshift_key=None,
):
    """Evaluate redshift-weighted weights to use to generate a
    rates-independent classifier.

    The redshift range is divided into logarithmically-spaced bins. Each class
    is given the same weights in each bin so that the rates information in the
    training set doesn't affect the classification. A classifier trained using
    these weights will produce a "rates-independent" classification.

    The redshift bins to use are set using a logarithmic range between
    min_redshift and max_redshift with a total of num_bins. Any objects that
    spill out of these bins are included in the first and last bins. A separate
    bin is included for galactic objects at redshift exactly 0.

    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to evaluate weights on.
    class_weights : dict (optional)
        Weights to use for each class. If not set, equal weights are assumed
        for each class.
    group_key : str (optional)
        If set, the group of each object will be loaded using group_key as the
        key in the dataset's metadata. The weights will be calculated
        independently for each group. This can be useful if there are multiple
        very different survey strategies in the same dataset, all of which have
        their own selection efficiencies. By default,
        settings['redshift_weighting_group_key'] will be used.
    min_redshift : float (optional)
        The minimum redshift bin to use. By default,
        settings['redshift_weighting_min_redshift'] will be used.
    max_redshift : float (optional)
        The maximum redshift bin to use. By default,
        settings['redshift_weighting_max_redshift'] will be used.
    num_bins : int (optional)
        The number of redshift bins to use. By default,
        settings['redshift_weighting_num_bins'] will be used.
    min_bin_count : int (optional)
        The minimum number of counts in each redshift bin. Tis is used to avoid
        having poorly sampled objects in the training set blow up the metric.
        By default, settings['redshift_weighting_min_bin_count'] will be used.
    redshift_key : str (optional)
        The key to use for determining the redshift. When training a
        classifier, this should typically be the spectroscopic redshift of the
        host galaxy because that is the measured "true" redshift for real
        samples. When evaluating on simulated data without spectroscopic
        redshifts, this might need to be changed to the true simulated
        redshift. By default, settings['redshift_weighting_redshift_key'] will
        be used.

    Returns
    -------
    weights : `pandas.Series`
        The weights that should be used for classification.
    """
    if group_key is None:
        group_key = settings["redshift_weighting_group_key"]
    if min_redshift is None:
        min_redshift = settings["redshift_weighting_min_redshift"]
    if max_redshift is None:
        max_redshift = settings["redshift_weighting_max_redshift"]
    if num_bins is None:
        num_bins = settings["redshift_weighting_num_bins"]
    if min_bin_count is None:
        min_bin_count = settings["redshift_weighting_min_bin_count"]
    if redshift_key is None:
        redshift_key = settings["redshift_weighting_redshift_key"]

    use_metadata = dataset.metadata

    # Create the initial bin range
    redshift_bins = np.logspace(
        np.log10(min_redshift), np.log10(max_redshift), num_bins + 1
    )

    # Replace the first and last bins with very small and large numbers to
    # effectively extend them to infinity.
    redshift_bins[0] = 1e-99
    redshift_bins[-1] = 1e99

    # Add in a bin for galactic objects at redshifts of exactly 0
    redshift_bins = np.hstack([-1e99, redshift_bins])

    # Figure out which redshift bin each object falls in.
    redshift_indices = np.searchsorted(redshift_bins, use_metadata[redshift_key]) - 1

    # Figure out how many different classes there are, and create a mapping for
    # them.
    object_classes = use_metadata["class"]
    class_names = np.unique(object_classes)
    class_map = {class_name: i for i, class_name in enumerate(class_names)}
    class_indices = [class_map[i] for i in object_classes]

    # Figure out how many different groups there are, and create a mapping for
    # them.
    if group_key is not None:
        groups = use_metadata[group_key]
        group_names = np.unique(groups)
        group_map = {group_name: i for i, group_name in enumerate(group_names)}
        group_indices = [group_map[i] for i in groups]
    else:
        group_names = ["default"]
        group_indices = np.zeros(len(use_metadata), dtype=int)

    # Count how many objects are in each bin.
    counts = np.zeros((len(group_names), len(redshift_bins) - 1, len(class_names)))
    for group_index, redshift_index, class_index in zip(
        group_indices, redshift_indices, class_indices
    ):
        counts[group_index, redshift_index, class_index] += 1

    total_counts = np.sum(counts)

    # Count how many extragalactic bins are actually populated. This is
    # used to set the scales so that they roughly match what we have
    # for the non-redshift-weighted metric. Note that the metric evaluation
    # won't be affected by this scale since for each class we divide by the
    # total weights of that class. However, this scaling is necessary if we
    # want to use the same hyperparameters for classification. For galactic
    # objects, we don't need to do anything because all of the observations end
    # up in the same bin. For extragalactic objects, we need to take into
    # account the fact that the objects are now split up between many different
    # bins. Get an estimate of how many bins are populated, and apply that to
    # the data.
    num_extgal_bins = np.sum(counts[:, 1:, :] > 1e-4 * total_counts)
    class_extgal_counts = np.sum(np.sum(counts[:, 1:, :], axis=0), axis=0)
    class_gal_counts = np.sum(counts[:, 0, :], axis=0)
    extgal_mask = class_extgal_counts > class_gal_counts
    num_extgal_classes = np.sum(extgal_mask)
    extgal_scale = num_extgal_bins / num_extgal_classes

    # Add a floor to the counts in each redshift bin to avoid absurdly heigh
    # weights.
    floor_counts = np.clip(counts, min_bin_count, None)

    # Calculate the weights for each bin using the floored counts.
    weights = total_counts / floor_counts

    # Rescale the weights for extragalactic classes.
    weights[:, :, extgal_mask] /= extgal_scale

    # If class_weights is set, rescale the weights for each class.
    if class_weights is not None:
        for class_idx, class_name in enumerate(class_names):
            weights[:, :, class_idx] *= class_weights[class_name]

    # Calculate the weights for each object
    object_weights = weights[group_indices, redshift_indices, class_indices]
    object_weights = pd.Series(object_weights, index=use_metadata.index)

    return object_weights


class Classifier:
    """Classifier used to classify the different objects in a dataset.

    Parameters
    ----------
    name : str
        The name of the classifier.
    """

    def __init__(self, name):
        self.name = name

    def train(self, dataset):
        """Train the classifier on a dataset

        This needs to be implemented in subclasses.

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset to use for training.
        """
        raise NotImplementedError

    def predict(self, dataset):
        """Generate predictions for a dataset

        This needs to be implemented in subclasses.

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset to generate predictions for.

        Returns
        -------
        predictions : :class:`pandas.DataFrame`
            A pandas Series with the predictions for each class.
        """
        raise NotImplementedError

    @property
    def path(self):
        """Get the path to where a classifier should be stored on disk"""
        return get_classifier_path(self.name)

    def write(self, overwrite=False):
        """Write a trained classifier to disk

        Parameters
        ----------
        name : str
            A unique name used to identify the classifier.
        overwrite : bool (optional)
            If a classifier with the same name already exists on disk and this
            is True, overwrite it. Otherwise, raise an AvocadoException.
        """
        import pickle

        path = self.path

        # Make the containing directory if it doesn't exist yet.
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

        # Handle if the file already exists.
        if os.path.exists(path):
            if overwrite:
                logger.warning("Overwriting %s..." % path)
                os.remove(path)
            else:
                raise AvocadoException("Dataset %s already exists! Can't write." % path)

        # Write the classifier to a pickle file
        with open(path, "wb") as output_file:
            pickle.dump(self, output_file)

    @classmethod
    def load(cls, name):
        """Load a classifier that was previously saved to disk

        Parameters
        ----------
        name : str
            A unique name used to identify the classifier to load.
        """
        import pickle

        path = get_classifier_path(name)

        # Write the classifier to a pickle file
        with open(path, "rb") as input_file:
            classifier = pickle.load(input_file)

        return classifier


class LightGBMClassifier(Classifier):
    """Feature based classifier using LightGBM to classify objects.

    This uses a weighted multi-class logarithmic loss that normalizes for the
    total counts of each class. This classifier is optimized for the metric
    used in the PLAsTiCC Kaggle challenge.

    Parameters
    ----------
    featurizer : :class:`Featurizer`
        The featurizer to use to select features for classification.
    class_weights : dict (optional)
        Weights to use for each class. If not set, equal weights are assumed
        for each class.
    weighting_function : function (optional)
        Function to use to evaluate weights. By default,
        `evaluate_weights_flat` is used which normalizes the weights for each
        class so that their overall weight matches the one set by
        class_weights. Within each class, `evaluate_weights_flat` gives all
        objects equal weights. Any weights function can be used here as long as
        it has the same signature as `evaluate_weights_flat`.
    """

    def __init__(
        self,
        name,
        featurizer,
        class_weights=None,
        weighting_function=evaluate_weights_flat,
    ):
        super().__init__(name)

        self.featurizer = featurizer
        self.class_weights = class_weights
        self.weighting_function = weighting_function

    def train(self, dataset, num_folds=None, random_state=None, **kwargs):
        """Train the classifier on a dataset

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset to use for training.
        num_folds : int (optional)
            The number of folds to use. Default: settings['num_folds']
        random_state : int (optional)
            The random number initializer to use for splitting the folds.
            Default: settings['fold_random_state']
        **kwargs
            Additional parameters to pass to the LightGBM classifier.
        """
        features = dataset.select_features(self.featurizer)

        # Label the folds
        folds = dataset.label_folds(num_folds, random_state)
        num_folds = np.max(folds) + 1

        object_weights = self.weighting_function(dataset, self.class_weights)
        object_classes = dataset.metadata["class"]
        classes = np.unique(object_classes)

        importances = pd.DataFrame()
        predictions = pd.DataFrame(
            -1 * np.ones((len(object_classes), len(classes))),
            index=dataset.metadata.index,
            columns=classes,
        )

        classifiers = []

        for fold in range(num_folds):
            print("Training fold %d." % fold)
            train_mask = folds != fold
            validation_mask = folds == fold

            train_features = features[train_mask]
            train_classes = object_classes[train_mask]
            train_weights = object_weights[train_mask]

            validation_features = features[validation_mask]
            validation_classes = object_classes[validation_mask]
            validation_weights = object_weights[validation_mask]

            classifier = fit_lightgbm_classifier(
                train_features,
                train_classes,
                train_weights,
                validation_features,
                validation_classes,
                validation_weights,
                **kwargs
            )

            validation_predictions = classifier.predict_proba(
                validation_features, num_iteration=classifier.best_iteration_
            )

            predictions[validation_mask] = validation_predictions

            importance = pd.DataFrame()
            importance["feature"] = features.columns
            importance["gain"] = classifier.feature_importances_
            importance["fold"] = fold + 1
            importances = pd.concat([importances, importance], axis=0, sort=False)

            classifiers.append(classifier)

        # Statistics on out-of-sample predictions
        total_logloss = weighted_multi_logloss(
            object_classes,
            predictions,
            object_weights=object_weights,
            class_weights=self.class_weights,
        )
        unweighted_total_logloss = weighted_multi_logloss(
            object_classes, predictions, class_weights=self.class_weights
        )
        print("Weighted log-loss:")
        print("    With object weights:    %.5f" % total_logloss)
        print("    Without object weights: %.5f" % unweighted_total_logloss)

        # Original sample only (no augments)
        if "reference_object_id" in dataset.metadata:
            original_mask = dataset.metadata["reference_object_id"].isnull()
            original_logloss = weighted_multi_logloss(
                object_classes[original_mask],
                predictions[original_mask],
                object_weights=object_weights[original_mask],
                class_weights=self.class_weights,
            )
            unweighted_original_logloss = weighted_multi_logloss(
                object_classes[original_mask],
                predictions[original_mask],
                class_weights=self.class_weights,
            )
            print("Original un-augmented dataset weighted log-loss:")
            print("    With object weights:    %.5f" % original_logloss)
            print("    Without object weights: %.5f" % unweighted_original_logloss)

        self.importances = importances
        self.train_predictions = predictions
        self.train_classes = object_classes
        self.classifiers = classifiers

        return classifiers

    def predict(self, dataset):
        """Generate predictions for a dataset

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset to generate predictions for.

        Returns
        -------
        predictions : :class:`pandas.DataFrame`
            A pandas Series with the predictions for each class.
        """
        features = dataset.select_features(self.featurizer)

        predictions = 0

        for classifier in tqdm(self.classifiers, desc="Classifier", dynamic_ncols=True):
            fold_scores = classifier.predict_proba(
                features, raw_score=True, num_iteration=classifier.best_iteration_
            )

            exp_scores = np.exp(fold_scores)

            fold_predictions = exp_scores / np.sum(exp_scores, axis=1)[:, None]
            predictions += fold_predictions

        predictions /= len(self.classifiers)

        predictions = pd.DataFrame(
            predictions, index=features.index, columns=self.train_predictions.columns
        )

        return predictions


def fit_lightgbm_classifier(
    train_features,
    train_classes,
    train_weights,
    validation_features,
    validation_classes,
    validation_weights,
    **kwargs
):
    """Fit a LightGBM classifier

    Parameters
    ----------
    train_features : `pandas.DataFrame`
        The features of the training objects.
    train_classes : `pandas.Series`
        The classes of the training objects.
    train_weights : `pandas.Series`
        The weights of the training objects.
    validation_features : `pandas.DataFrame`
        The features of the validation objects.
    validation_classes : `pandas.Series`
        The classes of the validation objects.
    validation_weights : `pandas.Series`
        The weights of the validation objects.
    **kwargs
        Additional parameters to pass to the LightGBM classifier.

    Returns
    -------
    classifier : `lightgbm.LGBMClassifier`
        The fitted LightGBM classifier
    """
    import lightgbm as lgb

    lgb_params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": len(np.unique(train_classes)),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "colsample_bytree": 0.5,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_split_gain": 10.0,
        "min_child_weight": 2000.0,
        "n_estimators": 5000,
        "silent": -1,
        "verbose": -1,
        "max_depth": 7,
        "num_leaves": 50,
    }

    lgb_params.update(kwargs)

    fit_params = {"verbose": 100, "sample_weight": train_weights}

    fit_params["eval_set"] = [(validation_features, validation_classes)]
    fit_params["early_stopping_rounds"] = 50
    fit_params["eval_sample_weight"] = [validation_weights]

    classifier = lgb.LGBMClassifier(**lgb_params)
    classifier.fit(train_features, train_classes, **fit_params)

    return classifier


def weighted_multi_logloss(
    true_classes,
    predictions,
    object_weights=None,
    class_weights=None,
    return_object_contributions=False,
):
    """Evaluate a weighted multi-class logloss function.

    Parameters
    ----------
    true_classes : `pandas.Series`
        A pandas series with the true class for each object
    predictions : `pandas.DataFrame`
        A pandas data frame with the predicted probabilities of each class for
        every object. There should be one column for each class.
    object_weights : dict (optional)
        The weights to use for each object. These are used to weight objects
        within a given class. The overall class weights will be normalized to
        the values set by class_weights. If not specified, flat weights are
        used.
    class_weights : dict (optional)
        The weights to use for each class. If not specified, flat weights are
        assumed for each class.
    return_object_contributions : bool (optional)
        If True, return a pandas Series with the individual contributions from
        each object. Otherwise, return the sum over all classes (default).

    Returns
    -------
    logloss : float or `pandas.Series`
        By default, return the weighted multi-class logloss over all classes.
        If return_object_contributions is True, this returns a pandas Series
        with the individual contributions to the logloss from each object
        instead.
    """
    object_loglosses = pd.Series(
        1e10 * np.ones(len(true_classes)), index=true_classes.index
    )

    sum_class_weights = 0

    for class_name in np.unique(true_classes):
        class_mask = true_classes == class_name
        class_count = np.sum(class_mask)

        if object_weights is not None:
            class_object_weights = object_weights[class_mask]
        else:
            class_object_weights = np.ones(class_count)

        if class_weights is not None:
            class_weight = class_weights.get(class_name, 1)
        else:
            class_weight = 1

        if class_weight == 0:
            # No weight for this class, ignore it.
            object_loglosses[class_mask] = 0
            continue

        if class_name not in predictions.columns:
            raise AvocadoException(
                "No predictions available for class %s! Either compute them "
                "or set the weight for that class to 0." % class_name
            )

        class_predictions = predictions[class_name][class_mask]

        class_loglosses = (
            -class_weight
            * class_object_weights
            * np.log(class_predictions)
            / np.sum(class_object_weights)
        )

        object_loglosses[class_mask] = class_loglosses

        sum_class_weights += class_weight

    object_loglosses /= sum_class_weights

    if return_object_contributions:
        return object_loglosses
    else:
        return np.sum(object_loglosses)
