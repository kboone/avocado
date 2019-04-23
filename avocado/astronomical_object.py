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
        - host_photometric_redshift: The photometric redshift of the object's
          host galaxy.
        - host_photometric_redshift_error: The error on the photometric
          redshift of the object's host galaxy.
        - host_spectroscopic_redshift: The spectroscopic redshift of the
          object's host galaxy.

        For training data objects, the following keys are assumed to exist in
        the metadata:
        - redshift: The true redshift of the object.
        - class: The true class label of the object.

    observations : pandas.DataFrame
        Observations of the object's light curve. This should be a pandas
        DataFrame with at least the following columns:

        - time: The time of each observation.
        - band: The band used for the observation.
        - flux: The measured flux value of the observation.
        - flux_error: The flux measurement uncertainty of the observation.
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

    def fit_gp(self, subtract_background=True, fix_scale=False, verbose=False,
               start_length_scale=20.):
        """Fit a Gaussian Process model to the light curve.

        Parameters
        ----------
        subtract_background : bool
            If True, a background subtraction routine is applied to the
            lightcurve before fitting the GP. Otherwise, the flux values are
            used as-is.
        fix_scale : bool
            If True, the scale is fixed to an initial estimate. If False, the
            scale is a free fit parameter (default).
        verbose : bool
            If True, output additional debugging information.
        start_length_scale : float
            The initial length scale to use for the fit.

        Returns
        -------
        gaussian_process : function
            A Gaussian process conditioned on the object's lightcurve. This is
            a wrapper around the george `predict` method with the object flux
            fixed.
        gp_data : dict
            A dictionary containing all of the information needed to build the
            Gaussian process.
        """

        fluxes = self.observations['flux']
        flux_errs = self.observations['flux_err']

        scale = fluxes[np.argmax(fluxes / (flux_errs + 1e-5))]


        # GP kernel. We use a 2-dimensional Matern kernel to model the
        # transient. The kernel amplitude is fixed to a fraction of the maximum
        # value in the data, and the kernel width in the wavelength direction
        # is also fixed. We fit for the kernel width in the time direction as
        # different transients evolve on very different time scales.
        kernel = ((0.2*gp_data['scale'])**2 *
                  kernels.Matern32Kernel([guess_length_scale**2, 6000**2],
                                         ndim=2))

        # print(kernel.get_parameter_names())
        if fix_scale:
            kernel.freeze_parameter('k1:log_constant')
        kernel.freeze_parameter('k2:metric:log_M_1_1')

        gp = george.GP(kernel)

        if verbose:
            print(kernel.get_parameter_dict())

        x_data = np.vstack([gp_data['times'], gp_data['wavelengths']]).T

        gp.compute(x_data, gp_data['flux_errs'])

        fluxes = gp_data['fluxes']

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(fluxes)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(fluxes)

        # print(np.exp(gp.get_parameter_vector()))

        bounds = [(0, np.log(1000**2))]
        if not fix_scale:
            bounds = [(-30, 30)] + bounds

        fit_result = minimize(
            neg_ln_like,
            gp.get_parameter_vector(),
            jac=grad_neg_ln_like,
            # bounds=[(-30, 30), (0, 10), (0, 5)],
            # bounds=[(0, 10), (0, 5)],
            bounds=bounds,
            # bounds=[(-30, 30), (0, np.log(1000**2))],
            # options={'ftol': 1e-4}
        )

        if not fit_result.success:
            print("GP Fit failed!")

        # print(-gp.log_likelihood(fluxes))
        # print(np.exp(fit_result.x))

        gp.set_parameter_vector(fit_result.x)

        if verbose:
            print(fit_result)
            print(kernel.get_parameter_dict())

        # Add results of the GP fit to the gp_data dictionary.
        gp_data['fit_parameters'] = fit_result.x

        # Build a GP function that is preconditioned on the known data.
        cond_gp = partial(gp.predict, gp_data['fluxes'])

        return cond_gp, gp_data

