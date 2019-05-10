"""Utility functions to interact with the PLAsTiCC dataset"""

from astropy.cosmology import FlatLambdaCDM
import numpy as np
import pandas as pd
import os

from .dataset import Dataset
from .utils import settings, AvocadoException, logger

from .augment import Augmentor


def update_plasticc_names(dataset_kind, metadata, observations=None):
    """Rename columns in PLAsTiCC tables to follow the avocado naming scheme.

    Parameters
    ----------
    dataset_kind : str {'training', 'test'}

    metadata : pandas.DataFrame
        Original metadata

    observations : pandas.DataFrame (optional)
        Original observations DataFrame

    Returns
    -------
    renamed_metadata : pandas.DataFrame
        metadata DataFrame with renamed columns to follow the avocado
        naming scheme.

    renamed_observations : pandas.DataFrame
        observations DataFrame with renamed columns to follow the avocado
        naming scheme. This is only returned if observations is not None.
    """

    # Rename columns in the metadata table to match the avocado standard.
    metadata_name_map = {
        'target': 'category',
        'hostgal_photoz_err': 'host_photoz_error',
        'hostgal_photoz': 'host_photoz',
        'hostgal_specz': 'host_specz',
    }
    metadata.rename(metadata_name_map, axis=1, inplace=True)

    # The true redshift is the host spectroscopic redshift for the PLAsTiCC
    # training set.
    if dataset_kind == 'training':
        metadata['redshift'] = metadata['host_specz']

    # Explicitly set a galactic/extragalactic flag.
    metadata['galactic'] = metadata['host_photoz'] == 0.

    if observations is None:
        return metadata

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


def load_plasticc_training():
    """Load the PLAsTiCC training set.
    
    Returns
    =======
    training_dataset : :class:`Dataset`
        The PLAsTiCC training dataset.
    """
    data_directory = settings['data_directory']

    observations_path = os.path.join(data_directory, 'training_set.csv')
    observations = pd.read_csv(observations_path)

    metadata_path = os.path.join(data_directory, 'training_set_metadata.csv')
    metadata = pd.read_csv(metadata_path)

    metadata, observations = update_plasticc_names('training', metadata,
                                                   observations)

    # Create a Dataset object
    dataset = Dataset('plasticc_training', metadata, observations)

    return dataset


def load_plasticc_test():
    """Load the metadata of the full PLAsTiCC test set.

    Only the metadata is loaded, not the individual observations. The
    individual observations can't all fit in memory at the same time on normal
    computers.

    Returns
    =======
    test_dataset : :class:`Dataset`
        The PLAsTiCC test dataset (metadata only).
    """
    data_directory = settings['data_directory']

    metadata_path = os.path.join(data_directory, 'test_set_metadata.csv')
    metadata = pd.read_csv(metadata_path)

    metadata = update_plasticc_names('test', metadata)

    # Create a Dataset object
    dataset = Dataset('plasticc_test', metadata)

    return dataset


class PlasticcAugmentor(Augmentor):
    """Implementation of an Augmentor for the PLAsTiCC dataset"""
    def __init__(self):
        super().__init__()

        self._test_dataset = None
        self._photoz_reference = None

        # Reverse engineered cosmology used to generate PLAsTiCC dataset
        self.cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

        # Load the photo-z model
        self._load_photoz_reference()

    def _load_test_dataset(self):
        """Load the full PLAsTiCC test dataset to use as a reference for
        augmentation.

        The metadata is cached as self._test_dataset. Only the metadata is
        loaded.

        Returns
        =======
        test_dataset : :class:`Dataset`
            The test dataset loaded with metadata only.
        """
        if self._test_dataset is None:
            self._test_dataset = load_plasticc_test()

        return self._test_dataset

    def _load_photoz_reference(self):
        """Load the full PLAsTiCC dataset as a reference for photo-z
        estimation.

        This reads the test set and extracts all of the photo-zs and true
        redshifts. The results are cached as self._photoz_reference.

        Returns
        =======
        photoz_reference numpy ndarray
            A Nx3 array with reference photo-zs for each entry with a spec-z in
            the test set. The columns are spec-z, photo-z and photo-z error.
        """
        if self._photoz_reference is None:
            logger.info("Loading photoz reference...")
            test_dataset = self._load_test_dataset()

            cut = test_dataset.metadata['host_specz'] > 0
            cut_metadata = test_dataset.metadata[cut]

            result = np.vstack([cut_metadata['host_specz'],
                                cut_metadata['host_photoz'],
                                cut_metadata['host_photoz_error']]).T

            self._photoz_reference = result

        return self._photoz_reference

    def _simulate_photoz(self, redshift):
        """Simulate the photoz determination for a lightcurve using the test
        set as a reference.

        I apply the observed differences between photo-zs and spec-zs directly
        to the new redshifts. This does not capture all of the intricacies of
        photo-zs, but it does ensure that we cover all of the available
        parameter space with at least some simulations.

        Parameters
        ----------
        redshift : float
            The new true redshift of the object.

        Returns
        -------
        host_photoz : float
            The simulated photoz of the host.

        host_photoz_error : float
            The simulated photoz error of the host.
        """
        photoz_reference = self._load_photoz_reference()

        while True:
            ref_idx = np.random.choice(len(photoz_reference))
            ref_specz, ref_photoz, ref_photoz_err = photoz_reference[ref_idx]

            # Randomly choose the order for the difference. Degeneracies work
            # both ways, so even if we only see specz=0.2 -> photoz=3.0 in the
            # data, the reverse also happens, but we can't get spec-zs at z=3
            # so we don't see this.
            new_diff = (ref_photoz - ref_specz) * np.random.choice([-1, 1])

            # Apply the difference, and make sure that the photoz is > 0.
            new_photoz = redshift + new_diff
            if new_photoz < 0:
                continue

            # Add some noise to the error so that the classifier can't focus in
            # on it.
            new_photoz_err = ref_photoz_err * np.random.normal(1, 0.05)

            break

        return new_photoz, new_photoz_err

    def _augment_redshift(self, reference_object, augmented_metadata):
        """Choose a new redshift and simulate the photometric redshift for an
        augmented object

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        augmented_metadata : dict
            The augmented metadata to add the new redshift too. This will be
            updated in place.
        """
        # Choose a new redshift.
        if reference_object.metadata['galactic']:
            # Galactic object, redshift stays the same
            augmented_metadata['redshift'] = 0
            augmented_metadata['host_specz'] = 0
            augmented_metadata['host_photoz'] = 0
            augmented_metadata['host_photoz_error'] = 0

            # Choose a factor (in magnitudes) to change the brightness by
            augmented_metadata['augment_brightness'] = (
                np.random.normal(0.5, 0.5)
            )
        else:
            # Choose a new redshift based on the reference template redshift.
            template_redshift = reference_object.metadata['redshift']

            # First, we limit the redshift range as a multiple of the original
            # redshift. We avoid making templates too much brighter because
            # the lower redshift templates will be left with noise that is
            # unrealistic. We also avoid going to too high of a relative
            # redshift because the templates there will be too faint to be
            # detected and the augmentor will waste a lot of time without being
            # able to actually generate a template.
            min_redshift = 0.95 * template_redshift
            max_redshift = 5 * template_redshift

            # Second, for high-redshift objects, we add a constraint to make
            # sure that we aren't evaluating the template at wavelengths where
            # the GP extrapolation is unreliable.
            max_redshift = np.min(
                [max_redshift, 1.5 * (1 + template_redshift) - 1]
            )

            # Choose new redshift from a log-uniform distribution over the
            # allowable redshift range.
            aug_redshift = np.exp(np.random.uniform(
                np.log(min_redshift), np.log(max_redshift)
            ))

            # Simulate a new photometric redshift
            aug_photoz, aug_photoz_error = self._simulate_photoz(aug_redshift)
            aug_distmod = self.cosmology.distmod(aug_photoz).value

            augmented_metadata['redshift'] = aug_redshift
            augmented_metadata['host_specz'] = aug_redshift
            augmented_metadata['host_photoz'] = aug_photoz
            augmented_metadata['host_photoz_error'] = aug_photoz_error
            augmented_metadata['distmod'] = aug_distmod

    def augment_metadata(self, reference_object):
        """Generate new metadata for the augmented object.

        This method needs to be implemented in survey-specific subclasses of
        this class. The new redshift, photoz, coordinates, etc. should be
        chosen in this method.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        Returns
        =======
        augmented_metadata : dict
            The augmented metadata
        """
        augmented_metadata = reference_object.metadata.copy()

        # Choose a new redshift.
        self._augment_redshift(reference_object, augmented_metadata)

        # Choose whether the new object will be in the DDF or not.
        if reference_object.metadata['ddf']:
            # Most observations are WFD observations, so generate more of
            # those. Thee DDF and WFD samples are effectively completely
            # different, so this ratio doesn't really matter.
            augmented_metadata['ddf'] = np.random.rand() > 0.8
        else:
            # If the reference wasn't a DDF observation, can't simulate a DDF
            # observation.
            augmented_metadata['ddf'] = False

        # Smear the mwebv value a bit so that it doesn't uniquely identify
        # points. I leave the position on the sky unchanged (ra, dec, etc.).
        # Don't put any of those variables directly into the classifier!
        augmented_metadata['mwebv'] *= np.random.normal(1, 0.1)

        return augmented_metadata
