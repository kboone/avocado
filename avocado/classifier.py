import numpy as np
import os
import pandas as pd

from .settings import settings

def get_classifier_path(tag):
    """Get the path to where a classifier should be stored on disk

    Parameters
    ----------
    tag : str
        The unique tag for the classifier.
    """
    classifier_directory = settings['classifier_directory']
    classifier_path = os.path.join(classifier_directory,
                                   'classifier_%s.pkl' % tag)

    return classifier_path


class Classifier():
    """Classifier used to classify the different objects in a dataset."""
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

    def write(self, tag, overwrite=False):
        """Write a trained classifier to disk

        Parameters
        ----------
        tag : str
            A unique tag used to identify the classifier.
        overwrite : bool (optional)
            If a classifier with the same tag already exists on disk and this
            is True, overwrite it. Otherwise, raise an AvocadoException.
        """
        import pickle

        path = get_classifier_path(tag)

        # Make the containing directory if it doesn't exist yet.
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

        # Handle if the file already exists.
        if os.path.exists(path):
            if overwrite:
                logger.warning("Overwriting %s..." % path)
                os.remove(path)
            else:
                raise AvocadoException(
                    "Dataset %s already exists! Can't write." % path
                )

        # Write the classifier to a pickle file
        with open(path, 'wb') as output_file:
            pickle.dump(self, output_file)

    @classmethod
    def load(cls, tag):
        """Load a classifier that was previously saved to disk

        Parameters
        ----------
        tag : str
            A unique tag used to identify the classifier to load.
        """
        import pickle

        path = get_classifier_path(tag)

        # Write the classifier to a pickle file
        with open(path, 'rb') as input_file:
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
    """
    def __init__(self, featurizer, class_weights=None):
        self.featurizer = featurizer
        self.class_weights = class_weights

    def evaluate_weights(self, dataset):
        """Evaluate the weights to use for classification on a dataset.

        The weights are set to normalize each class to have same weight. If
        self.class_weights is set, those weights are applied after
        normalization.

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset to evaluate weights on.

        Returns
        -------
        weights : `pandas.Series`
            The weights that should be used for classification.
        """
        object_classes = dataset.metadata['category']
        class_counts = object_classes.value_counts()

        norm_class_weights = {}
        for class_name, class_count in class_counts.items():
            if self.class_weights is not None:
                class_weight = self.class_weights[class_name]
            else:
                class_weight = 1

            norm_class_weights[class_name] = (
                class_weight * len(object_classes) / class_count
            )

        weights = object_classes.map(norm_class_weights)

        return weights

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
        **kwargs
            Additional parameters to pass to the LightGBM classifier.
        """
        features = dataset.select_features(self.featurizer)

        # Label the folds
        folds = dataset.label_folds(num_folds, random_state)
        num_folds = np.max(folds) + 1

        weights = self.evaluate_weights(dataset)

        object_classes = dataset.metadata['category']
        classes = np.unique(object_classes)

        importances = pd.DataFrame()
        predictions = pd.DataFrame(
            -1 * np.ones((len(object_classes), len(classes))),
            index=dataset.metadata.index,
            columns=classes
        )

        classifiers = []

        for fold in range(num_folds):
            print("Training fold %d." % fold)
            train_mask = folds != fold
            validation_mask = folds == fold

            train_features = features[train_mask]
            train_classes = object_classes[train_mask]
            train_weights = weights[train_mask]

            validation_features = features[validation_mask]
            validation_classes = object_classes[validation_mask]
            validation_weights = weights[validation_mask]

            classifier = fit_lightgbm_classifier(
                train_features, train_classes, train_weights,
                validation_features, validation_classes, validation_weights,
                **kwargs
            )

            validation_predictions = classifier.predict_proba(
                validation_features, num_iteration=classifier.best_iteration_
            )

            predictions[validation_mask] = validation_predictions

            importance = pd.DataFrame()
            importance['feature'] = features.columns
            importance['gain'] = classifier.feature_importances_
            importance['fold'] = fold + 1
            importances = pd.concat([importances, importance], axis=0,
                                    sort=False)

            classifiers.append(classifier)

        # Statistics on out-of-sample predictions
        total_logloss = weighted_multi_logloss(
            object_classes, predictions, self.class_weights
        )
        print('Total weighted log-loss: %.5f ' % total_logloss)

        # Original sample only (no augments)
        if 'reference_object_id' in dataset.metadata:
            original_mask = dataset.metadata['reference_object_id'].isnull()
            original_logloss = weighted_multi_logloss(
                object_classes[original_mask],
                predictions[original_mask],
                self.class_weights
            )
            print('Original weighted log-loss: %.5f ' % original_logloss)

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

        for classifier in self.classifiers:
            fold_scores = classifier.predict_proba(
                features, raw_score=True,
                num_iteration=classifier.best_iteration_
            )

            exp_scores = np.exp(fold_scores)

            fold_predictions = exp_scores / np.sum(exp_scores, axis=1)[:, None]
            predictions += fold_predictions

        predictions /= len(self.classifiers)

        predictions = pd.DataFrame(predictions, index=features.index,
                                   columns=self.train_predictions.columns)

        return predictions


def fit_lightgbm_classifier(train_features, train_classes, train_weights,
                            validation_features, validation_classes,
                            validation_weights, **kwargs):
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
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': len(np.unique(train_classes)),
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'colsample_bytree': .5,
        'reg_alpha': 0.,
        'reg_lambda': 0.,
        'min_split_gain': 10.,
        'min_child_weight': 2000.,
        'n_estimators': 5000,
        'silent': -1,
        'verbose': -1,
        'max_depth': 7,
        'num_leaves': 50,
    }

    lgb_params.update(kwargs)

    fit_params = {
        'verbose': 100,
        'sample_weight': train_weights,
    }

    fit_params['eval_set'] = [(validation_features, validation_classes)]
    fit_params['early_stopping_rounds'] = 50
    fit_params['eval_sample_weight'] = [validation_weights]

    classifier = lgb.LGBMClassifier(**lgb_params)
    classifier.fit(train_features, train_classes, **fit_params)

    return classifier


def weighted_multi_logloss(true_classes, predictions, class_weights=None,
                           return_class_contributions=False):
    """Evaluate a weighted multi-class logloss function.

    Parameters
    ----------
    true_classes : `pandas.Series`
        A pandas series with the true class for each object
    predictions : `pandas.DataFrame`
        A pandas data frame with the predicted probabilities of each class for
        every object. There should be one column for each class.
    class_weights : dict (optional)
        The weights to use for each class. If not specified, flat weights are
        assumed for each class.
    return_class_contributions : bool (optional)
        If True, return a pandas Series with the contributions from each
        class. Otherwise, return the sum over all classes (default).

    Returns
    -------
    logloss : float or `pandas.Series`
        By default, return the weighted multi-class logloss over all classes.
        If return_class_contributions is True, this returns a pandas Series
        with the individual contributions to the logloss from each class
        instead.
    """
    class_loglosses = []
    sum_weights = 0

    for class_name in predictions.columns:
        class_mask = true_classes == class_name

        class_count = np.sum(class_mask)
        class_predictions = predictions[class_name][class_mask]

        class_logloss = -np.sum(np.log(class_predictions)) / class_count

        if class_weights is not None:
            weight = class_weights.get(class_name, 1)
        else:
            weight = 1

        class_loglosses.append(weight * class_logloss)
        sum_weights += weight

    class_loglosses = pd.Series(
        np.array(class_loglosses) / sum_weights,
        index=predictions.columns
    )

    if return_class_contributions:
        return class_loglosses
    else:
        return np.sum(class_loglosses)
