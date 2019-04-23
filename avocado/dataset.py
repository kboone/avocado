import numpy as np

from sklearn.model_selection import StratifiedKFold

from .astronomical_object import AstronomicalObject
from .logging import logger
from .settings import settings

class Dataset():
    """Class representing a dataset of many astronomical objects.
    
    Parameters
    ----------
    metadata : pandas.DataFrame
        DataFrame where each row is the metadata for an object in the dataset.
        See :class:`AstronomicalObject` for details.

    observations : pandas.DataFrame
        Observations of all of the objects' light curves. See
        :class:`AstronomicalObject` for details.

    name : str
        Name of the dataset. This will be used to determine the filenames of
        various outputs such as computed features and predictions.
    """
    def __init__(self, name, metadata, observations=None):
        """Create a new Dataset from a set of metadata and observations"""

        # Use object_id as the index
        metadata = metadata.set_index('object_id')
        metadata.sort_index(inplace=True)

        self.name = name
        self.metadata = metadata

        # Label folds for training datasets
        if 'class' in self.metadata:
            self.label_folds()

        # Load each astronomical object in the dataset.
        objects = []
        for object_id, object_observations in \
                observations.groupby('object_id'):
            object_metadata = metadata.loc[object_id]
            new_object = AstronomicalObject(object_metadata,
                                            object_observations)
            objects.append(new_object)

        self.objects = np.array(objects)

    def label_folds(self):
        """Separate the dataset into groups for k-folding

        This is only applicable to training datasets that have assigned
        classes.

        The number of folds is set by the `num_folds` settings parameter.

        This needs to happen before augmentation to avoid leakage, so augmented
        datasets and similar datasets should already have the folds set.
        """
        if 'class' not in self.metadata:
            logger.warn("Dataset %s does not have labeled classes! Can't "
                        "separate into folds." % self.name)
            return

        num_folds = settings['num_folds']

        if 'fold' in self.metadata:
            # Warn if the fold count doesn't match.
            data_num_folds = np.max(self.metadata['fold']) + 1
            if data_num_folds != num_folds:
                logger.warn("Using %d preset folds in dataset instead of "
                            "%d requested." % (data_num_folds, num_folds))
            return

        # Label folds
        classes = self.metadata['class']
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                random_state=1)
        kfold_indices = -1 * np.ones(len(classes), dtype=int)
        for idx, (fold_train, fold_val) in \
                enumerate(folds.split(classes, classes)):
            kfold_indices[fold_val] = idx
        self.metadata['fold'] = kfold_indices
