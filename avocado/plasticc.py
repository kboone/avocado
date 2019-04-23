"""Utility functions to interact with the PLAsTiCC dataset"""

import pandas as pd
import os

from . import Dataset, settings

def update_plasticc_names(metadata, observations):
    """Rename columns in PLAsTiCC tables to follow the avocado naming scheme.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Original metadata

    observations : pandas.DataFrame
        Original observations DataFrame

    Returns
    -------
    renamed_metadata : pandas.DataFrame
        metadata DataFrame with renamed columns to follow the avocado
        naming scheme.

    renamed_observations : pandas.DataFrame
        observations DataFrame with renamed columns to follow the avocado
        naming scheme.
    """

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

    metadata.rename({'target': 'class'}, axis=1, inplace=True)

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

    metadata, observations = update_plasticc_names(metadata, observations)

    # Create a Dataset object
    dataset = Dataset('plasticc_trainng', metadata, observations)

    return dataset
