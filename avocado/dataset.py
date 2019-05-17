import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

from .astronomical_object import AstronomicalObject
from .utils import logger, AvocadoException, write_dataframe, \
    read_dataframes, read_dataframe
from .settings import settings

class Dataset():
    """Class representing a dataset of many astronomical objects.

    Parameters
    ----------
    name : str
        Name of the dataset. This will be used to determine the filenames of
        various outputs such as computed features and predictions.
    metadata : pandas.DataFrame
        DataFrame where each row is the metadata for an object in the dataset.
        See :class:`AstronomicalObject` for details.
    observations : pandas.DataFrame
        Observations of all of the objects' light curves. See
        :class:`AstronomicalObject` for details.
    objects : list
        A list of :class:`AstronomicalObject` instances. Either this or
        observations can be specified but not both.
    chunk : int (optional)
        If the dataset was loaded in chunks, this indicates the chunk number.
    num_chunks : int (optional)
        If the dataset was loaded in chunks, this is the total number of chunks
        used.
    """
    def __init__(self, name, metadata, observations=None, objects=None,
                 chunk=None, num_chunks=None):
        """Create a new Dataset from a set of metadata and observations"""
        # Make copies of everything so that we don't mess anything up.
        metadata = metadata.copy()
        if observations is not None:
            observations = observations.copy()

        self.name = name
        self.metadata = metadata
        self.chunk = chunk
        self.num_chunks = num_chunks

        if observations is None:
            self.objects = objects
        else:
            # Load each astronomical object in the dataset.
            self.objects = np.zeros(len(self.metadata), dtype=object)
            self.objects[:] = None
            meta_dicts = self.metadata.to_dict('records')
            for object_id, object_observations in \
                    observations.groupby('object_id'):
                meta_index = self.metadata.index.get_loc(object_id)

                # Make sure that every object_id only appears once in the
                # metadata. OTherwise we have a corrupted dataset that we can't
                # handle.
                if type(meta_index) != int:
                    raise AvocadoException(
                        "Error: found multiple metadata entries for "
                        "object_id=%s! Can't handle." % object_id
                    )

                object_metadata = meta_dicts[meta_index]
                object_metadata['object_id'] = object_id
                new_object = AstronomicalObject(object_metadata,
                                                object_observations)

                self.objects[meta_index] = new_object

        # Other variables that will be populated by various methods.
        self.raw_features = None
        self.features = None

    @property
    def path(self):
        """Return the path to where this dataset should lie on disk"""
        data_directory = settings['data_directory']
        data_path = os.path.join(data_directory, self.name + '.h5')

        return data_path

    def get_raw_features_path(self, tag=None):
        """Return the path to where the raw features for this dataset should
        lie on disk

        Parameters
        ----------
        tag : str (optional)
            The version of the raw features to use. By default, this will use
            settings['features_tag'].
        """
        if tag is None:
            tag = settings['features_tag']

        features_directory = settings['features_directory']

        features_filename = '%s_%s.h5' % (tag, self.name)
        features_path = os.path.join(features_directory, features_filename)

        return features_path

    @classmethod
    def load(cls, name, metadata_only=False, chunk=None, num_chunks=None,
             **kwargs):
        """Load a dataset that has been saved in HDF5 format in the data
        directory.

        For an example of how to create such a dataset, see
        `scripts/download_plasticc.py`.

        The dataset can optionally be loaded in chunks. To do this, pass chunk
        and num_chunks to this method. See `read_dataframes` for details.

        Parameters
        ----------
        name : str
            The name of the dataset to load
        metadata_only : bool (optional)
            If False (default), the observations are loaded. Otherwise, only
            the metadata is loaded. This is useful for very large datasets.
        chunk : int (optional)
            If set, load the dataset in chunks. chunk specifies the chunk
            number to load. This is a zero-based index.
        num_chunks : int (optional)
            The total number of chunks to use.
        **kwargs
            Additional arguments to `read_dataframes`

        Returns
        -------
        dataset : :class:`Dataset`
            The loaded dataset.
        """
        data_directory = settings['data_directory']
        data_path = os.path.join(data_directory, name + '.h5')

        if not os.path.exists(data_path):
            raise AvocadoException("Couldn't find dataset %s!" % name)

        if metadata_only:
            keys = ['metadata']
        else:
            keys = ['metadata', 'observations']

        dataframes = read_dataframes(data_path, keys, chunk=chunk,
                                     num_chunks=num_chunks, **kwargs)

        # Create a Dataset object
        dataset = cls(name, *dataframes, chunk=chunk, num_chunks=num_chunks)

        return dataset

    @classmethod
    def from_objects(cls, name, objects, **kwargs):
        """Load a dataset from a list of AstronomicalObject instances.

        Parameters
        ----------
        objects : list
            A list of AstronomicalObject instances.
        name : str
            The name of the dataset.
        **kwargs
            Additional arguments to pass to Dataset()

        Returns
        -------
        dataset : :class:`Dataset`
            The loaded dataset.
        """
        # Pull the metadata out of the objects
        metadata = pd.DataFrame([i.metadata for i in objects])
        metadata.set_index('object_id', inplace=True)

        # Load the new dataset.
        dataset = cls(name, metadata, objects=objects, **kwargs)

        return dataset

    def label_folds(self):
        """Separate the dataset into groups for k-folding

        This is only applicable to training datasets that have assigned
        categories.

        The number of folds is set by the `num_folds` settings parameter.

        If the dataset is an augmented dataset, we ensure that the
        augmentations of the same object stay in the same fold.
        """
        print("TODO: KEEP AUGMENTS IN SAME FOLD!")
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
        object_id : str (optional)
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
                index = self.metadata.index.get_loc(str(object_id))
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
        **kwargs
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

    def write(self, **kwargs):
        """Write the dataset out to disk.

        The dataset will be stored in the data directory using the dataset's
        name.

        Parameters
        ----------
        **kwargs
            Additional arguments to be passed to `utils.write_dataframe`
        """
        # Pull out the observations from every object
        observations = []
        for obj in self.objects:
            object_observations = obj.observations
            object_observations['object_id'] = obj.metadata['object_id']
            observations.append(object_observations)
        observations = pd.concat(observations, ignore_index=True, sort=False)

        write_dataframe(
            self.path, self.metadata, 'metadata', chunk=self.chunk,
            num_chunks=self.num_chunks, **kwargs
        )

        write_dataframe(
            self.path, observations, 'observations', index_chunk_column=False,
            chunk=self.chunk, num_chunks=self.num_chunks, **kwargs
        )

    def extract_raw_features(self, featurizer):
        """Extract raw features from the dataset.

        The raw features are saved as `self.raw_features`.

        Parameters
        ----------
        featurizer : :class:`Featurizer`
            The featurizer that will be used to calculate the features.

        Returns
        -------
        raw_features : pandas.DataFrame
            The extracted raw features.
        """
        list_raw_features = []
        object_ids = []
        for obj in tqdm(self.objects):
            obj_features = featurizer.extract_raw_features(obj)
            list_raw_features.append(obj_features.values())
            object_ids.append(obj.metadata['object_id'])

        # Pull the keys off of the last extraction. They should be the same for
        # every set of features.
        keys = obj_features.keys()

        raw_features = pd.DataFrame(list_raw_features, index=object_ids,
                                    columns=keys)
        raw_features.index.name = 'object_id'

        self.raw_features = raw_features

        return raw_features

    def select_features(self, featurizer):
        """Select features from the dataset for classification.

        This method assumes that the raw features have already been extracted
        for this dataset and are available with `self.raw_features`. Use
        `extract_raw_features` to calculate these from the data directly, or
        `load_features` to recover features that were previously stored on
        disk.

        The features are saved as `self.features`.

        Parameters
        ----------
        featurizer : :class:`Featurizer`
            The featurizer that will be used to select the features.

        Returns
        -------
        features : pandas.DataFrame
            The selected features.
        """
        if self.raw_features is None:
            raise AvocadoException(
                "Must calculate raw features before selecting features!"
            )

        features = featurizer.select_features(self.raw_features)

        self.features = features

        return features

    def write_raw_features(self, tag=None, **kwargs):
        """Write the raw features out to disk.

        The features will be stored in the features directory using the
        dataset's name and the given features tag.

        Parameters
        ----------
        tag : str (optional)
            The tag for this version of the features. By default, this will use
            settings['features_tag'].
        **kwargs
            Additional arguments to be passed to `utils.write_dataframe`
        """
        raw_features_path = self.get_raw_features_path(tag=tag)

        write_dataframe(
            raw_features_path,
            self.raw_features,
            'raw_features',
            chunk=self.chunk,
            num_chunks=self.num_chunks,
            **kwargs
        )

    def load_raw_features(self, tag=None, **kwargs):
        """Load the raw features from disk.

        Parameters
        ----------
        tag : str (optional)
            The version of the raw features to use. By default, this will use
            settings['features_tag'].

        Returns
        -------
        raw_features : pandas.DataFrame
            The extracted raw features.
        """
        raw_features_path = self.get_raw_features_path(tag=tag)

        self.raw_features = read_dataframe(
            raw_features_path,
            'raw_features',
            chunk=self.chunk,
            num_chunks=self.num_chunks,
            **kwargs
        )

        return self.raw_features
