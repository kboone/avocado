import pandas as pd

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

    def _get_gp_data(self, object_meta, object_data, fix_background=True):
        times = []
        fluxes = []
        bands = []
        flux_errs = []
        wavelengths = []

        # The zeropoints were arbitrarily set from the first image. Pick the
        # 20th percentile of all observations in each channel as a new
        # zeropoint. This has good performance when there are supernova-like
        # bursts in the image, even if they are quite wide.
        # UPDATE: when picking the 20th percentile, observations with just
        # noise get really messed up. Revert back to the median for now and see
        # if that helps. It doesn't really matter if supernovae go slightly
        # negative...
        # UPDATE 2: most of the objects of interest are short-lived in time.
        # The only issue with the background occurs when there was flux from
        # the transient in the reference image. To deal with this, look at the
        # last observations and see if they are negative (indicating that the
        # reference has additional flux in it). If so, then update the
        # background level. Otherwise, leave the background at the reference
        # level.
        for passband in range(num_passbands):
            band_data = object_data[object_data['passband'] == passband]
            if len(band_data) == 0:
                # No observations in this band
                continue

            # Use a biweight location to estimate the background
            ref_flux = biweight_location(band_data['flux'])

            for idx, row in band_data.iterrows():
                times.append(row['mjd'] - start_mjd)
                flux = row['flux']
                if fix_background:
                    flux = flux - ref_flux
                bands.append(passband)
                wavelengths.append(band_wavelengths[passband])
                fluxes.append(flux)
                flux_errs.append(row['flux_err'])

        times = np.array(times)
        bands = np.array(bands)
        wavelengths = np.array(wavelengths)
        fluxes = np.array(fluxes)
        flux_errs = np.array(flux_errs)

        # Guess the scale based off of the highest signal-to-noise point.
        # Sometimes the edge bands are pure noise and can have large
        # insignificant points. Add epsilon to this calculation to avoid divide
        # by zero errors for model fluxes that have 0 error.
        scale = fluxes[np.argmax(fluxes / (flux_errs + 1e-5))]

        gp_data = {
            'meta': object_meta,
            'times': times,
            'bands': bands,
            'scale': scale,
            'wavelengths': wavelengths,
            'fluxes': fluxes,
            'flux_errs': flux_errs,
        }

        return gp_data
