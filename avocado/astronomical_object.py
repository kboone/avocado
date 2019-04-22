import pandas as pd

class AstronomicalObject(object):
    """Class representing an astronomical object.

    An astronomical object has both metadata describing its global properties,
    and observations of its light curve.

    Parameters
    ----------
    metadata : dict-like
        Metadata for this object. This is represented using a dict
        internally, and must be able to be cast to a dict. Any keys and
        information are allowed. Various functions assume that the
        following keys exist in the metadata:

        - object_id: A unique ID for the object.
        - hostgal_photoz: The photometric redshift of the object's host
          galaxy.
        - hostgal_photoz_err: The error on the photometric redshift of the
          object's host galaxy.
        - hostgal_specz: The spectroscopic redshift of the object's host
          galaxy.
        - class: The true class label of the object (only available for the
          training data).

    observations : DataFrame
        Observations of the object's light curve. This should be a pandas
        DataFrame with at least the following columns:

        - mjd: The MJD date of each observation.
        - passband: The passband used for the observation.
        - flux: The measured flux value of the observation.
        - flux_err: The flux measurement uncertainty of the observation.
    """

    def __init__(self, metadata, observations):
        """Create a new AstronomicalObject"""
        self.metadata = metadata
        self.observations = observations
