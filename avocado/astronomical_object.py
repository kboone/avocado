import pandas as pd
from astropy.stats import biweight_location

class AstronomicalObject():
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

    observations : pandas.DataFrame
        Observations of the object's light curve. This should be a pandas
        DataFrame with at least the following columns:

        - mjd: The MJD date of each observation.
        - band: The band used for the observation.
        - flux: The measured flux value of the observation.
        - flux_err: The flux measurement uncertainty of the observation.
    """
    def __init__(self, metadata, observations):
        """Create a new AstronomicalObject"""
        self.metadata = dict(metadata)
        self.observations = observations

    @property
    def bands(self):
        """Return a list of bands that this object has observations in."""
        return np.unique(self.observations['band'])

    def subtract_background(self):
        """Subtract the background levels from each band.

        The background levels are estimated using a biweight location
        estimator. This estimator will calculate a robust estimate of the
        background level for objects that have short-lived light curves, and it
        will return something like the median flux level for periodic or
        continuous light curves.

        Returns
        -------
        subtracted_observations : pandas.DataFrame
            A modified version of the observations DataFrame with the
            background level removed.
        """
        subtracted_observations = self.observations.copy()

        for band in self.bands:
            mask = self.observations['band'] == band
            band_data = self.observations['mask']

            # Use a biweight location to estimate the background
            ref_flux = biweight_location(band_data['flux'])

            subtracted_observations['flux', mask] -= ref_flux

        return subtracted_observations

    def plot_light_curve(self, data_only=False):
        """Plot the object's light curve"""
        result = self.predict_gp(*args, **kwargs)

        plt.figure()

        for band in range(num_passbands):
            cut = result['bands'] == band
            color = band_colors[band]
            plt.errorbar(result['times'][cut], result['fluxes'][cut],
                         result['flux_errs'][cut], fmt='o', c=color,
                         markersize=5, label=band_names[band])

            if data_only:
                continue

            plt.plot(result['pred_times'], result['pred'][band], c=color)

            if kwargs.get('uncertainties', False):
                # Show uncertainties with a shaded band
                pred = result['pred'][band]
                err = np.sqrt(result['pred_var'][band])
                plt.fill_between(result['pred_times'], pred-err, pred+err,
                                 alpha=0.2, color=color)

        plt.legend()

        plt.xlabel('Time (days)')
        plt.ylabel('Flux')
        plt.tight_layout()

    def __repr__(self):
        return "AstronomicalObject(object_id=%s)" % self.metadata['object_id']

