"""Utility functions to interact with the PLAsTiCC dataset"""

import pandas as pd
import os

from .dataset import Dataset
from .utils import settings, AvocadoException

def update_plasticc_names(metadata, observations, dataset_kind):
    """Rename columns in PLAsTiCC tables to follow the avocado naming scheme.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Original metadata

    observations : pandas.DataFrame
        Original observations DataFrame

    dataset_kind : str {'training'}

    Returns
    -------
    renamed_metadata : pandas.DataFrame
        metadata DataFrame with renamed columns to follow the avocado
        naming scheme.

    renamed_observations : pandas.DataFrame
        observations DataFrame with renamed columns to follow the avocado
        naming scheme.
    """

    # Rename columns in the metadata table to match the avocado standard.
    metadata_name_map = {
        'target': 'class',
        'hostgal_specz': 'host_spectroscopic_redshift',
        'hostgal_photoz': 'host_photometric_redshift',
        'hostgal_photoz_err': 'host_photometric_redshift_error',
    }
    metadata.rename(metadata_name_map, axis=1, inplace=True)

    # The true redshift is the host spectroscopic redshift for the PLAsTiCC
    # training set.
    if dataset_kind == 'training':
        metadata['redshift'] = metadata['host_spectroscopic_redshift']

    # Replace the passband number with a string representing the LSST band.
    band_map = {
        0: 'lsstu',
        1: 'lsstg',
        2: 'lsstr',
        3: 'lssti',
        4: 'lsstz',
        5: 'lssty',
    }

    observations['band'] = observations['passband'].map(band_map)
    observations.drop('passband', axis=1, inplace=True)

    # Rename columns in the observations table to match the avocado standard.
    observations_name_map = {
        'mjd': 'time',
        'flux_err': 'flux_error',
    }
    observations.rename(observations_name_map, axis=1, inplace=True)

    return metadata, observations

def load_training_set():
    """Load the PLAsTiCC training set.
    
    Returns
    -------
    training_dataset : Dataset
        The loaded Dataset object
    """
    data_directory = settings['data_directory']

    observations_path = os.path.join(data_directory, 'training_set.csv')
    observations = pd.read_csv(observations_path)

    metadata_path = os.path.join(data_directory, 'training_set_metadata.csv')
    metadata = pd.read_csv(metadata_path)

    metadata, observations = update_plasticc_names(metadata, observations,
                                                   'training')

    # Create a Dataset object
    dataset = Dataset('plasticc_trainng', metadata, observations)

    return dataset
