from astropy.stats import biweight_location
from functools import partial
import george
from george import kernels
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .instruments import (
    get_band_central_wavelength,
    get_band_plot_color,
    get_band_plot_marker,
)
from .utils import logger


class AstronomicalObject:
    """An astronomical object, with metadata and a lightcurve.

    An astronomical object has both metadata describing its global properties,
    and observations of its light curve.

    Parameters
    ----------
    metadata : dict-like
        Metadata for this object. This is represented using a dict
        internally, and must be able to be cast to a dict. Any keys and
        information are allowed. Various functions assume that the
        following keys exist in the metadata:

        - object_id: A unique ID for the object. This will be stored as a
          string internally.
        - galactic: Whether or not the object is in the Milky Way galaxy or
          not.
        - host_photoz: The photometric redshift of the object's host galaxy.
        - host_photoz_error: The error on the photometric redshift of the
          object's host galaxy.
        - host_specz: The spectroscopic redshift of the object's host galaxy.

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

        self._default_gaussian_process = None

    def __repr__(self):
        return f"{type(self).__name__}(object_id={self.metadata['object_id']})"

    @property
    def bands(self):
        """Return a list of bands that this object has observations in

        Returns
        -------
        bands : numpy.array
            A list of bands, ordered by their central wavelength.
        """
        unsorted_bands = np.unique(self.observations["band"])
        sorted_bands = np.array(sorted(unsorted_bands, key=get_band_central_wavelength))
        return sorted_bands

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
            mask = self.observations["band"] == band
            band_data = self.observations[mask]

            # Use a biweight location to estimate the background
            ref_flux = biweight_location(band_data["flux"])

            subtracted_observations.loc[mask, "flux"] -= ref_flux

        return subtracted_observations

    def preprocess_observations(self, subtract_background=True, **kwargs):
        """Apply preprocessing to the observations.

        This function is intended to be used to transform the raw observations
        table into one that can actually be used for classification. For now,
        all that this step does is apply background subtraction.

        Parameters
        ----------
        subtract_background : bool (optional)
            If True (the default), a background subtraction routine is applied
            to the lightcurve before fitting the GP. Otherwise, the flux values
            are used as-is.
        kwargs : dict
            Additional keyword arguments. These are ignored. We allow
            additional keyword arguments so that the various functions that
            call this one can be called with the same arguments, even if they
            don't actually use them.

        Returns
        -------
        preprocessed_observations : pandas.DataFrame
            The preprocessed observations that can be used for further
            analyses.
        """
        if subtract_background:
            preprocessed_observations = self.subtract_background()
        else:
            preprocessed_observations = self.observations

        return preprocessed_observations

    def fit_gaussian_process(
        self,
        fix_scale=False,
        verbose=False,
        guess_length_scale=20.0,
        **preprocessing_kwargs
    ):
        """Fit a Gaussian Process model to the light curve.

        We use a 2-dimensional Matern kernel to model the transient. The kernel
        width in the wavelength direction is fixed. We fit for the kernel width
        in the time direction as different transients evolve on very different
        time scales.

        Parameters
        ----------
        fix_scale : bool (optional)
            If True, the scale is fixed to an initial estimate. If False
            (default), the scale is a free fit parameter.
        verbose : bool (optional)
            If True, output additional debugging information.
        guess_length_scale : float (optional)
            The initial length scale to use for the fit. The default is 20
            days.
        preprocessing_kwargs : kwargs (optional)
            Additional preprocessing arguments that are passed to
            `preprocess_observations`.

        Returns
        -------
        gaussian_process : function
            A Gaussian process conditioned on the object's lightcurve. This is
            a wrapper around the george `predict` method with the object flux
            fixed.
        gp_observations : pandas.DataFrame
            The processed observations that the GP was fit to. This could have
            effects such as background subtraction applied to it.
        gp_fit_parameters : list
            A list of the resulting GP fit parameters.
        """
        gp_observations = self.preprocess_observations(**preprocessing_kwargs)

        fluxes = gp_observations["flux"]
        flux_errors = gp_observations["flux_error"]

        wavelengths = gp_observations["band"].map(get_band_central_wavelength)
        times = gp_observations["time"]

        # Use the highest signal-to-noise observation to estimate the scale. We
        # include an error floor so that in the case of very high
        # signal-to-noise observations we pick the maximum flux value.
        signal_to_noises = np.abs(fluxes) / np.sqrt(
            flux_errors ** 2 + (1e-2 * np.max(fluxes)) ** 2
        )
        scale = np.abs(fluxes[signal_to_noises.idxmax()])

        kernel = (0.5 * scale) ** 2 * kernels.Matern32Kernel(
            [guess_length_scale ** 2, 6000 ** 2], ndim=2
        )

        if fix_scale:
            kernel.freeze_parameter("k1:log_constant")
        kernel.freeze_parameter("k2:metric:log_M_1_1")

        gp = george.GP(kernel)

        guess_parameters = gp.get_parameter_vector()

        if verbose:
            print(kernel.get_parameter_dict())

        x_data = np.vstack([times, wavelengths]).T

        gp.compute(x_data, flux_errors)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(fluxes)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(fluxes)

        bounds = [(0, np.log(1000 ** 2))]
        if not fix_scale:
            bounds = [(guess_parameters[0] - 10, guess_parameters[0] + 10)] + bounds

        fit_result = minimize(
            neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, bounds=bounds
        )

        if fit_result.success:
            gp.set_parameter_vector(fit_result.x)
        else:
            # Fit failed. Print out a warning, and use the initial guesses for
            # fit parameters. This only really seems to happen for objects
            # where the lightcurve is almost entirely noise.
            logger.warn(
                "GP fit failed for %s! Using guessed GP parameters. "
                "This is usually OK." % self
            )
            gp.set_parameter_vector(guess_parameters)

        if verbose:
            print(fit_result)
            print(kernel.get_parameter_dict())

        # Return the Gaussian process and associated data.
        gaussian_process = partial(gp.predict, fluxes)

        return gaussian_process, gp_observations, fit_result.x

    def get_default_gaussian_process(self):
        """Get the default Gaussian Process.

        This method calls fit_gaussian_process with the default arguments and
        caches its output so that multiple calls only require fitting the GP a
        single time.
        """
        if self._default_gaussian_process is None:
            gaussian_process, _, _ = self.fit_gaussian_process()
            self._default_gaussian_process = gaussian_process

        return self._default_gaussian_process

    def predict_gaussian_process(
        self, bands, times, uncertainties=True, fitted_gp=None, **gp_kwargs
    ):
        """Predict the Gaussian process in a given set of bands and at a given
        set of times.

        Parameters
        ==========
        bands : list(str)
            bands to predict the Gaussian process in.
        times : list or numpy.array of floats
            times to evaluate the Gaussian process at.
        uncertainties : bool (optional)
            If True (default), the GP uncertainties are computed and returned
            along with the mean prediction. If False, only the mean prediction
            is returned.
        fitted_gp : function (optional)
            By default, this function will perform the GP fit before doing
            predictions. If the GP fit has already been done, then the fitted
            GP function (returned by fit_gaussian_process) can be passed here
            instead to skip redoing the fit.
        gp_kwargs : kwargs (optional)
            Additional arguments that are passed to `fit_gaussian_process`.

        Returns
        =======
        predictions : numpy.array
            A 2-dimensional array with shape (len(bands), len(times))
            containing the Gaussian process mean flux predictions.
        prediction_uncertainties : numpy.array
            Only returned if uncertainties is True. This is an array with the
            same shape as predictions containing the Gaussian process
            uncertainty for the predictions.
        """
        if fitted_gp is not None:
            gp = fitted_gp
        else:
            gp, _, _ = self.fit_gaussian_process(**gp_kwargs)

        # Predict the Gaussian process band-by-band.
        predictions = []
        prediction_uncertainties = []

        for band in bands:
            wavelengths = np.ones(len(times)) * get_band_central_wavelength(band)
            pred_x_data = np.vstack([times, wavelengths]).T
            if uncertainties:
                band_pred, band_pred_var = gp(pred_x_data, return_var=True)
                prediction_uncertainties.append(np.sqrt(band_pred_var))
            else:
                band_pred = gp(pred_x_data, return_cov=False)
            predictions.append(band_pred)

        predictions = np.array(predictions)
        if uncertainties:
            prediction_uncertainties = np.array(prediction_uncertainties)
            return predictions, prediction_uncertainties
        else:
            return predictions

    def plot_light_curve(self, show_gp=True, verbose=False, axis=None, **kwargs):
        """Plot the object's light curve

        Parameters
        ----------
        show_gp : bool (optional)
            If True (default), the Gaussian process prediction is plotted along
            with the raw data.
        verbose : bool (optional)
            If True, print detailed information about the light curve and GP
            fit.
        axis : `matplotlib.axes.Axes` (optional)
            The matplotlib axis to plot to. If None, a new figure will be
            created.
        kwargs : kwargs (optional)
            Additional arguments. If show_gp is True, these are passed to
            `fit_gaussian_process`. Otherwise, these are passed to
            `preprocess_observations`.
        """
        if verbose:
            self.print_metadata()

        if show_gp:
            gp, observations, gp_fit_parameters = self.fit_gaussian_process(
                verbose=verbose, **kwargs
            )
        else:
            observations = self.preprocess_observations(**kwargs)

        # Figure out the times to plot. We go 10% past the edges of the
        # observations.
        min_time_obs = np.min(observations["time"])
        max_time_obs = np.max(observations["time"])
        border = 0.1 * (max_time_obs - min_time_obs)
        min_time = min_time_obs - border
        max_time = max_time_obs + border

        if show_gp:
            pred_times = np.arange(min_time, max_time + 1)

            predictions, prediction_uncertainties = self.predict_gaussian_process(
                self.bands, pred_times, fitted_gp=gp
            )

        if axis is None:
            figure, axis = plt.subplots()

        for band_idx, band in enumerate(self.bands):
            mask = observations["band"] == band
            band_data = observations[mask]
            color = get_band_plot_color(band)
            marker = get_band_plot_marker(band)

            axis.errorbar(
                band_data["time"],
                band_data["flux"],
                band_data["flux_error"],
                fmt="o",
                c=color,
                markersize=6,
                marker=marker,
                label=band,
            )

            if not show_gp:
                continue

            pred = predictions[band_idx]
            axis.plot(pred_times, pred, c=color)
            err = prediction_uncertainties[band_idx]

            if kwargs.get("uncertainties", True):
                # If they were calculated, show uncertainties with a shaded
                # band.
                axis.fill_between(
                    pred_times, pred - err, pred + err, alpha=0.2, color=color
                )

        axis.legend()

        axis.set_xlabel("Time (days)")
        axis.set_ylabel("Flux")
        axis.set_xlim(min_time, max_time)

        axis.figure.tight_layout()

    def print_metadata(self):
        """Print out the object's metadata in a nice format."""
        # Try to print out specific keys in a nice order. If these keys aren't
        # available, then we skip them. The rest of the keys are printed out in
        # a random order afterwards.
        ordered_keys = [
            "object_id",
            "class",
            "galactic",
            "fold",
            "redshift",
            "host_specz",
            "host_photoz",
            "host_photoz_error",
        ]
        for key in ordered_keys:
            if key in self.metadata:
                print("%20s: %s" % (key, self.metadata[key]))

        for key, value in self.metadata.items():
            if key in ordered_keys:
                continue
            print("%20s: %s" % (key, value))
