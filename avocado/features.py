class Featurizer:
    """Class used to extract features from objects."""

    def extract_raw_features(self, astronomical_object, return_model=False):
        """Extract raw features from an object

        Featurizing is slow, so the idea here is to extract a lot of different
        things, and then in `select_features` these features are postprocessed
        to select the ones that are actually fed into the classifier. This
        allows for rapid iteration of training on different feature sets. Note
        that the features produced by this method are often unsuitable for
        classification, and may include data leaks. Make sure that you
        understand what features can be used for real classification before
        making any changes.

        For now, there is no generic featurizer, so this must be implemented in
        survey-specific subclasses.

        Parameters
        ----------
        astronomical_object : :class:`AstronomicalObject`
            The astronomical object to featurize.
        return_model : bool
            If true, the light curve model is also returned. Defaults to False.

        Returns
        -------
        raw_features : dict
            The raw extracted features for this object.
        model : dict (optional)
            A dictionary with the light curve model in each band. This is only
            returned if return_model is set to True.
        """
        return NotImplementedError

    def select_features(self, raw_features):
        """Select features to use for classification

        This method should take a DataFrame or dictionary of raw features,
        produced by `featurize`, and output a list of processed features that
        can be fed to a classifier.

        Parameters
        ----------
        raw_features : pandas.DataFrame or dict
            The raw features extracted using `featurize`.

        Returns
        -------
        features : pandas.DataFrame or dict
            The processed features that can be fed to a classifier.
        """
        return NotImplementedError

    def extract_features(self, astronomical_object):
        """Extract features from an object.

        This method extracts raw features with `extract_raw_features`, and then
        selects the ones that should be used for classification with
        `select_features`. This method is just a wrapper around those two
        methods and is intended to be used as a shortcut for feature extraction
        on individual objects.

        Parameters
        ----------
        astronomical_object : :class:`AstronomicalObject`
            The astronomical object to featurize.

        Returns
        -------
        features : pandas.DataFrame or dict
            The processed features that can be fed to a classifier.
        """
        raw_features = self.extract_raw_features(astronomical_object)
        features = self.select_features(raw_features)

        return features
