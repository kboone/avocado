import numpy as np

from sklearn.model_selection import StratifiedKFold

from .astronomical_object import AstronomicalObject
from .utils import logger, AvocadoException
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
        if 'category' in self.metadata:
            self.label_folds()

        if observations is None:
            self.objects = None
        else:
            # Load each astronomical object in the dataset.
            objects = []
            meta_dicts = self.metadata.to_dict('records')
            for object_id, object_observations in \
                    observations.groupby('object_id'):
                meta_index = self.metadata.index.get_loc(object_id)
                object_metadata = meta_dicts[meta_index]
                object_metadata['object_id'] = object_id
                new_object = AstronomicalObject(object_metadata,
                                                object_observations)
                objects.append(new_object)

            self.objects = np.array(objects)

    def label_folds(self):
        """Separate the dataset into groups for k-folding

        This is only applicable to training datasets that have assigned
        categories.

        The number of folds is set by the `num_folds` settings parameter.

        This needs to happen before augmentation to avoid leakage, so augmented
        datasets and similar datasets should already have the folds set.
        """
        if 'category' not in self.metadata:
            logger.warn("Dataset %s does not have labeled categories! Can't "
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
        categories = self.metadata['category']
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True,
                                random_state=1)
        kfold_indices = -1 * np.ones(len(categories), dtype=int)
        for idx, (fold_train, fold_val) in \
                enumerate(folds.split(categories, categories)):
            kfold_indices[fold_val] = idx
        self.metadata['fold'] = kfold_indices

    def get_object(self, index=None, category=None, object_id=None):
        """Parse keywords to pull a specific object out of the dataset

        Parameters
        ==========
        index : int (optional)
            The index of the object in the dataset in the range
            [0, num_objects-1]. If a specific category is specified, then the
            index only counts objects of that category.
        category : int or str (optional)
            Filter for objects of a specific category. If this is specified,
            then index must also be specified.
        object_id : hashable (optional)
            Retrieve an object with this specific object_id. If index or
            category is specified, then object_id cannot also be specified.

        Returns
        =======
        astronomical_object : AstronomicalObject
            The object that was retrieved.
        """
        # Check to make sure that we have a valid object specification.
        base_error = "Error finding object! "
        if object_id is not None:
            if index is not None or category is not None:
                raise AvocadoException(
                    base_error + "If object_id is specified, can't also "
                    "specify index or category!"
                )

        if category is not None and index is None:
            raise AvocadoException(
                base_error + "Must specify index if category is specified!"
            )

        # Figure out the index to use.
        if category is not None:
            # Figure out the target object_id and use that to get the index.
            category_index = index
            category_meta = self.metadata[
                self.metadata['category'] == category]
            object_id = category_meta.index[category_index]

        if object_id is not None:
            try:
                index = self.metadata.index.get_loc(object_id)
            except KeyError:
                raise AvocadoException(
                    base_error + "No object with object_id=%s" % object_id
                )

        return self.objects[index]

    def _get_object(self, index=None, category=None, object_id=None, **kwargs):
        """Wrapper around get_object that returns unused kwargs.

        This function is used for the common situation of pulling an object out
        of the dataset and doing additional processing on it. The arguments
        used by get_object are removed from the arguments list, and the
        remainder are returned. See `get_object` for details of the parameters.
        
        Returns
        =======
        astronomical_object : AstronomicalObject
            The object that was retrieved.
        kwargs : dict
            Additional arguments passed to the function that weren't used.
        """
        return self.get_object(index, category, object_id), kwargs

    def plot_light_curve(self, *args, **kwargs):
        """Plot the light curve for an object in the dataset.

        See `get_object` for the various keywords that can be used to choose
        the object. Additional keywords are passed to
        `AstronomicalObject.plot()`
        """
        target_object, plot_kwargs = self._get_object(*args, **kwargs)
        target_object.plot_light_curve(**plot_kwargs)

    def plot_interactive(self):
        """Make an interactive plot of the light curves in the dataset.

        This requires the ipywidgets package to be set up, and has only been
        tested in jupyter-lab.
        """
        from ipywidgets import interact, IntSlider, Dropdown

        categories = {'' : None}
        for category in np.unique(self.metadata['category']):
            categories[category] = category

        idx_widget = IntSlider(min=0, max=1)
        category_widget = Dropdown(options=categories, index=0)

        def update_idx_range(*args):
            if category_widget.value is None:
                idx_widget.max = len(self.metadata) - 1
            else:
                idx_widget.max = np.sum(self.metadata['category'] ==
                                        category_widget.value) - 1

        category_widget.observe(update_idx_range, 'value')

        update_idx_range()

        interact(self.plot_light_curve, index=idx_widget,
                 category=category_widget, show_gp=True, uncertainties=True,
                 verbose=False, subtract_background=True)
