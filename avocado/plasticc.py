"""Implementation of avocado components for the PLAsTiCC dataset"""

import numpy as np
import os
import pandas as pd
from scipy.special import erf

from .augment import Augmentor
from .dataset import Dataset
from .features import Featurizer
from .settings import settings
from .utils import AvocadoException, logger


# PLAsTiCC specific parameters
pad = 100
plasticc_start_time = 59580
plasticc_end_time = 60675
plasticc_bands = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]

plasticc_kaggle_weights = {
    6: 1,
    15: 2,
    16: 1,
    42: 1,
    52: 1,
    53: 1,
    62: 1,
    64: 2,
    65: 1,
    67: 1,
    88: 1,
    90: 1,
    92: 1,
    95: 1,
    99: 2,
}
plasticc_flat_weights = {
    6: 1,
    15: 1,
    16: 1,
    42: 1,
    52: 1,
    53: 1,
    62: 1,
    64: 1,
    65: 1,
    67: 1,
    88: 1,
    90: 1,
    92: 1,
    95: 1,
    99: 0,
}


class PlasticcAugmentor(Augmentor):
    """Implementation of an Augmentor for the PLAsTiCC dataset"""

    def __init__(self):
        super().__init__()

        self._photoz_reference = None

        # Load the photo-z model
        self._load_photoz_reference()

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

            data_directory = settings["data_directory"]
            data_path = os.path.join(data_directory, "plasticc_test.h5")

            # When reading the metadata table, pandas will actually load all
            # columns even if only a few are specified. To avoid having a huge
            # memory footprint, load the metadata in chunks and pull out the
            # columns that we care about.
            chunksize = 10 ** 5

            use_metadata = []
            for chunk in pd.read_hdf(
                data_path, "metadata", mode="r", chunksize=chunksize
            ):
                cut = chunk["host_specz"] > 0
                use_metadata.append(
                    chunk[cut][["host_specz", "host_photoz", "host_photoz_error"]]
                )

            use_metadata = pd.concat(use_metadata)

            result = np.vstack(
                [
                    use_metadata["host_specz"],
                    use_metadata["host_photoz"],
                    use_metadata["host_photoz_error"],
                ]
            ).T

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
        if reference_object.metadata["galactic"]:
            # Galactic object, redshift stays the same
            augmented_metadata["redshift"] = 0
            augmented_metadata["host_specz"] = 0
            augmented_metadata["host_photoz"] = 0
            augmented_metadata["host_photoz_error"] = 0

            # Choose a factor (in magnitudes) to change the brightness by
            augmented_metadata["augment_brightness"] = np.random.normal(0.5, 0.5)
        else:
            # Choose a new redshift based on the reference template redshift.
            template_redshift = reference_object.metadata["redshift"]

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
            max_redshift = np.min([max_redshift, 1.5 * (1 + template_redshift) - 1])

            # Choose new redshift from a log-uniform distribution over the
            # allowable redshift range.
            aug_redshift = np.exp(
                np.random.uniform(np.log(min_redshift), np.log(max_redshift))
            )

            # Simulate a new photometric redshift
            aug_photoz, aug_photoz_error = self._simulate_photoz(aug_redshift)
            aug_distmod = self.cosmology.distmod(aug_photoz).value

            augmented_metadata["redshift"] = aug_redshift
            augmented_metadata["host_specz"] = aug_redshift
            augmented_metadata["host_photoz"] = aug_photoz
            augmented_metadata["host_photoz_error"] = aug_photoz_error
            augmented_metadata["augment_brightness"] = 0.0

    def _augment_metadata(self, reference_object):
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
        if reference_object.metadata["ddf"]:
            # Most observations are WFD observations, so generate more of
            # those. The DDF and WFD samples are effectively completely
            # different, so this ratio doesn't really matter.
            augmented_metadata["ddf"] = np.random.rand() > 0.8
        else:
            # If the reference wasn't a DDF observation, can't simulate a DDF
            # observation.
            augmented_metadata["ddf"] = False

        # Smear the mwebv value a bit so that it doesn't uniquely identify
        # points. I leave the position on the sky unchanged (ra, dec, etc.).
        # Don't put any of those variables directly into the classifier!
        augmented_metadata["mwebv"] *= np.random.normal(1, 0.1)

        return augmented_metadata

    def _choose_target_observation_count(self, augmented_metadata):
        """Choose the target number of observations for a new augmented light
        curve.

        We use a functional form that roughly maps out the number of
        observations in the PLAsTiCC test dataset for each of the DDF and WFD
        samples.

        Parameters
        ----------
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        target_observation_count : int
            The target number of observations in the new light curve.
        """
        if augmented_metadata["ddf"]:
            target_observation_count = int(np.random.normal(330, 30))
        else:
            # I estimate the distribution of number of observations in the
            # WFD regions with a mixture of 3 gaussian distributions.
            gauss_choice = np.random.choice(3, p=[0.05, 0.4, 0.55])
            if gauss_choice == 0:
                mu = 95
                sigma = 20
            elif gauss_choice == 1:
                mu = 115
                sigma = 8
            elif gauss_choice == 2:
                mu = 138
                sigma = 8
            target_observation_count = int(
                np.clip(np.random.normal(mu, sigma), 50, None)
            )

        return target_observation_count

    def _simulate_light_curve_uncertainties(self, observations, augmented_metadata):
        """Simulate the observation-related noise and detections for a light
        curve.

        For the PLAsTiCC dataset, we estimate the measurement uncertainties for
        each band with a lognormal distribution for both the WFD and DDF
        surveys. Those measurement uncertainties are added to the simulated
        observations.

        Parameters
        ----------
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process. These observations have model flux uncertainties listed
            that should be included in the final uncertainties.
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        observations : pandas.DataFrame
            The observations with uncertainties added.
        """
        # Make a copy so that we don't modify the original array.
        observations = observations.copy()

        if len(observations) == 0:
            # No data, skip
            return observations

        if augmented_metadata["ddf"]:
            band_noises = {
                "lsstu": (0.68, 0.26),
                "lsstg": (0.25, 0.50),
                "lsstr": (0.16, 0.36),
                "lssti": (0.53, 0.27),
                "lsstz": (0.88, 0.22),
                "lssty": (1.76, 0.23),
            }
        else:
            band_noises = {
                "lsstu": (2.34, 0.43),
                "lsstg": (0.94, 0.41),
                "lsstr": (1.30, 0.41),
                "lssti": (1.82, 0.42),
                "lsstz": (2.56, 0.36),
                "lssty": (3.33, 0.37),
            }

        # Calculate the new noise levels using a lognormal distribution for
        # each band.
        lognormal_parameters = []
        for band in observations["band"]:
            try:
                lognormal_parameters.append(band_noises[band])
            except KeyError:
                raise AvocadoException(
                    "Noise properties of band %s not known, add them in "
                    "PlasticcAugmentor._simulate_light_curve_uncertainties." % band
                )
        lognormal_parameters = np.array(lognormal_parameters)

        add_stds = np.random.lognormal(
            lognormal_parameters[:, 0], lognormal_parameters[:, 1]
        )

        noise_add = np.random.normal(loc=0.0, scale=add_stds)
        observations["flux"] += noise_add
        observations["flux_error"] = np.sqrt(
            observations["flux_error"] ** 2 + add_stds ** 2
        )

        return observations

    def _simulate_detection(self, observations, augmented_metadata):
        """Simulate the detection process for a light curve.

        We model the PLAsTiCC detection probabilities with an error function.
        I'm not entirely sure why this isn't deterministic. The full light
        curve is considered to be detected if there are at least 2 individual
        detected observations.

        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process.
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        observations : pandas.DataFrame
            The observations with the detected flag set.
        pass_detection : bool
            Whether or not the full light curve passes the detection thresholds
            used for the full sample.
        """
        s2n = np.abs(observations["flux"]) / observations["flux_error"]
        prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.0
        observations["detected"] = np.random.rand(len(s2n)) < prob_detected

        pass_detection = np.sum(observations["detected"]) >= 2

        return observations, pass_detection


class PlasticcFeaturizer(Featurizer):
    """Class used to extract features for the PLAsTiCC dataset."""

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

        This class implements a featurizer that is tuned to the PLAsTiCC
        dataset.

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
        from scipy.signal import find_peaks

        features = dict()

        # Fit the GP and produce an output model
        gp_start_time = plasticc_start_time - pad
        gp_end_time = plasticc_end_time + pad
        gp_times = np.arange(gp_start_time, gp_end_time + 1)
        gp, gp_observations, gp_fit_parameters = (
            astronomical_object.fit_gaussian_process()
        )
        gp_fluxes = astronomical_object.predict_gaussian_process(
            plasticc_bands, gp_times, uncertainties=False, fitted_gp=gp
        )

        times = gp_observations["time"]
        fluxes = gp_observations["flux"]
        flux_errors = gp_observations["flux_error"]
        bands = gp_observations["band"]
        s2ns = fluxes / flux_errors
        metadata = astronomical_object.metadata

        # Features from the metadata
        features["host_specz"] = metadata["host_specz"]
        features["host_photoz"] = metadata["host_photoz"]
        features["host_photoz_error"] = metadata["host_photoz_error"]
        features["ra"] = metadata["ra"]
        features["decl"] = metadata["decl"]
        features["mwebv"] = metadata["mwebv"]
        features["ddf"] = metadata["ddf"]

        # Count how many observations there are
        features["count"] = len(fluxes)

        # Features from GP fit parameters
        for i, fit_parameter in enumerate(gp_fit_parameters):
            features["gp_fit_%d" % i] = fit_parameter

        # Maximum fluxes and times.
        max_times = gp_start_time + np.argmax(gp_fluxes, axis=1)
        med_max_time = np.median(max_times)
        max_dts = max_times - med_max_time
        max_fluxes = np.array(
            [
                gp_fluxes[band_idx, time - gp_start_time]
                for band_idx, time in enumerate(max_times)
            ]
        )
        features["max_time"] = med_max_time
        for band, max_flux, max_dt in zip(plasticc_bands, max_fluxes, max_dts):
            features["max_flux_%s" % band] = max_flux
            features["max_dt_%s" % band] = max_dt

        # Minimum fluxes.
        min_fluxes = np.min(gp_fluxes, axis=1)
        for band, min_flux in zip(plasticc_bands, min_fluxes):
            features["min_flux_%s" % band] = min_flux

        # Calculate the positive and negative integrals of the lightcurve,
        # normalized to the respective peak fluxes. This gives a measure of the
        # "width" of the lightcurve, even for non-bursty objects.
        positive_widths = np.sum(np.clip(gp_fluxes, 0, None), axis=1) / max_fluxes
        negative_widths = np.sum(np.clip(gp_fluxes, None, 0), axis=1) / min_fluxes
        for band_idx, band_name in enumerate(plasticc_bands):
            features["positive_width_%s" % band_name] = positive_widths[band_idx]
            features["negative_width_%s" % band_name] = negative_widths[band_idx]

        # Calculate the total absolute differences of the lightcurve. For
        # supernovae, they typically go up and down a single time. Periodic
        # objects will have many more ups and downs.
        abs_diffs = np.sum(np.abs(gp_fluxes[:, 1:] - gp_fluxes[:, :-1]), axis=1)
        for band_idx, band_name in enumerate(plasticc_bands):
            features["abs_diff_%s" % band_name] = abs_diffs[band_idx]

        # Find times to fractions of the peak amplitude
        fractions = [0.8, 0.5, 0.2]
        for band_idx, band_name in enumerate(plasticc_bands):
            forward_times = find_time_to_fractions(gp_fluxes[band_idx], fractions)
            backward_times = find_time_to_fractions(
                gp_fluxes[band_idx], fractions, forward=False
            )
            for fraction, forward_time, backward_time in zip(
                fractions, forward_times, backward_times
            ):
                features["time_fwd_max_%.1f_%s" % (fraction, band_name)] = forward_time
                features["time_bwd_max_%.1f_%s" % (fraction, band_name)] = backward_time

        # Count the number of data points with significant positive/negative
        # fluxes
        thresholds = [-20, -10, -5, -3, 3, 5, 10, 20]
        for threshold in thresholds:
            if threshold < 0:
                count = np.sum(s2ns < threshold)
            else:
                count = np.sum(s2ns > threshold)
            features["count_s2n_%d" % threshold] = count

        # Count the fraction of data points that are "background", i.e. less
        # than a 3 sigma detection of something.
        features["frac_background"] = np.sum(np.abs(s2ns) < 3) / len(s2ns)

        for band_idx, band_name in enumerate(plasticc_bands):
            mask = bands == band_name
            band_fluxes = fluxes[mask]
            band_flux_errors = flux_errors[mask]

            # Sum up the total signal-to-noise in each band
            total_band_s2n = np.sqrt(np.sum((band_fluxes / band_flux_errors) ** 2))
            features["total_s2n_%s" % band_name] = total_band_s2n

            # Calculate percentiles of the data in each band.
            for percentile in (10, 30, 50, 70, 90):
                try:
                    val = np.percentile(band_fluxes, percentile)
                except IndexError:
                    val = np.nan
                features["percentile_%s_%d" % (band_name, percentile)] = val

        # Count the time delay between the first and last significant fluxes
        thresholds = [5, 10, 20]
        for threshold in thresholds:
            significant_times = times[np.abs(s2ns) > threshold]
            if len(significant_times) < 2:
                dt = -1
            else:
                dt = np.max(significant_times) - np.min(significant_times)
            features["time_width_s2n_%d" % threshold] = dt

        # Count how many data points are within a certain number of days of
        # maximum light. This provides some estimate of the robustness of the
        # determination of maximum light and rise/fall times.
        time_bins = [
            (-5, 5, "center"),
            (-20, -5, "rise_20"),
            (-50, -20, "rise_50"),
            (-100, -50, "rise_100"),
            (-200, -100, "rise_200"),
            (-300, -200, "rise_300"),
            (-400, -300, "rise_400"),
            (-500, -400, "rise_500"),
            (-600, -500, "rise_600"),
            (-700, -600, "rise_700"),
            (-800, -700, "rise_800"),
            (5, 20, "fall_20"),
            (20, 50, "fall_50"),
            (50, 100, "fall_100"),
            (100, 200, "fall_200"),
            (200, 300, "fall_300"),
            (300, 400, "fall_400"),
            (400, 500, "fall_500"),
            (500, 600, "fall_600"),
            (600, 700, "fall_700"),
            (700, 800, "fall_800"),
        ]
        diff_times = times - med_max_time
        for start, end, label in time_bins:
            mask = (diff_times > start) & (diff_times < end)

            # Count how many observations there are in the time bin
            count = np.sum(mask)
            features["count_max_%s" % label] = count

            if count == 0:
                bin_mean_fluxes = np.nan
                bin_std_fluxes = np.nan
            else:
                # Measure the GP flux level relative to the peak flux. We do
                # this by taking the median flux in each band and comparing it
                # to the peak flux.
                bin_start = np.clip(
                    int(med_max_time + start - gp_start_time), 0, len(gp_times)
                )
                bin_end = np.clip(
                    int(med_max_time + end - gp_start_time), 0, len(gp_times)
                )

                if bin_start == bin_end:
                    scale_gp_fluxes = np.nan
                    bin_mean_fluxes = np.nan
                    bin_std_fluxes = np.nan
                else:
                    scale_gp_fluxes = (
                        gp_fluxes[:, bin_start:bin_end] / max_fluxes[:, None]
                    )
                    bin_mean_fluxes = np.mean(scale_gp_fluxes)
                    bin_std_fluxes = np.std(scale_gp_fluxes)

            features["mean_max_%s" % label] = bin_mean_fluxes
            features["std_max_%s" % label] = bin_std_fluxes

        # Do peak detection on the GP output
        for positive in (True, False):
            for band_idx, band_name in enumerate(plasticc_bands):
                if positive:
                    band_flux = gp_fluxes[band_idx]
                    base_name = "peaks_pos_%s" % band_name
                else:
                    band_flux = -gp_fluxes[band_idx]
                    base_name = "peaks_neg_%s" % band_name
                peaks, properties = find_peaks(
                    band_flux, height=np.max(np.abs(band_flux) / 5.0)
                )
                num_peaks = len(peaks)

                features["%s_count" % base_name] = num_peaks

                sort_heights = np.sort(properties["peak_heights"])[::-1]
                # Measure the fractional height of the other peaks.
                for i in range(1, 3):
                    if num_peaks > i:
                        rel_height = sort_heights[i] / sort_heights[0]
                    else:
                        rel_height = np.nan
                    features["%s_frac_%d" % (base_name, (i + 1))] = rel_height

        if return_model:
            # Return the GP predictions along with the features.
            model = {}
            for idx, band in enumerate(plasticc_bands):
                model["%s" % band] = gp_fluxes[idx]
            model["time"] = gp_times
            model = pd.DataFrame(model).set_index("time")

            return features, model
        else:
            # Only return the features.
            return features

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
        rf = raw_features

        # Make a new dict or pandas DataFrame for the features. Everything is
        # agnostic about whether raw_features is a dict or a pandas DataFrame
        # and the output will be the same as the input.
        features = type(rf)()

        # Keys that we want to use directly for classification.
        copy_keys = ["host_photoz", "host_photoz_error"]

        for copy_key in copy_keys:
            features[copy_key] = rf[copy_key]

        features["length_scale"] = rf["gp_fit_1"]

        max_flux = rf["max_flux_lssti"]
        max_mag = -2.5 * np.log10(np.abs(max_flux))

        features["max_mag"] = max_mag

        features["pos_flux_ratio"] = rf["max_flux_lssti"] / (
            rf["max_flux_lssti"] - rf["min_flux_lssti"]
        )
        features["max_flux_ratio_red"] = np.abs(rf["max_flux_lssty"]) / (
            np.abs(rf["max_flux_lssty"]) + np.abs(rf["max_flux_lssti"])
        )
        features["max_flux_ratio_blue"] = np.abs(rf["max_flux_lsstg"]) / (
            np.abs(rf["max_flux_lssti"]) + np.abs(rf["max_flux_lsstg"])
        )

        features["min_flux_ratio_red"] = np.abs(rf["min_flux_lssty"]) / (
            np.abs(rf["min_flux_lssty"]) + np.abs(rf["min_flux_lssti"])
        )
        features["min_flux_ratio_blue"] = np.abs(rf["min_flux_lsstg"]) / (
            np.abs(rf["min_flux_lssti"]) + np.abs(rf["min_flux_lsstg"])
        )

        features["max_dt"] = rf["max_dt_lssty"] - rf["max_dt_lsstg"]

        features["positive_width"] = rf["positive_width_lssti"]
        features["negative_width"] = rf["negative_width_lssti"]

        features["time_fwd_max_0.5"] = rf["time_fwd_max_0.5_lssti"]
        features["time_fwd_max_0.2"] = rf["time_fwd_max_0.2_lssti"]

        features["time_fwd_max_0.5_ratio_red"] = rf["time_fwd_max_0.5_lssty"] / (
            rf["time_fwd_max_0.5_lssty"] + rf["time_fwd_max_0.5_lssti"]
        )
        features["time_fwd_max_0.5_ratio_blue"] = rf["time_fwd_max_0.5_lsstg"] / (
            rf["time_fwd_max_0.5_lsstg"] + rf["time_fwd_max_0.5_lssti"]
        )
        features["time_fwd_max_0.2_ratio_red"] = rf["time_fwd_max_0.2_lssty"] / (
            rf["time_fwd_max_0.2_lssty"] + rf["time_fwd_max_0.2_lssti"]
        )
        features["time_fwd_max_0.2_ratio_blue"] = rf["time_fwd_max_0.2_lsstg"] / (
            rf["time_fwd_max_0.2_lsstg"] + rf["time_fwd_max_0.2_lssti"]
        )

        features["time_bwd_max_0.5"] = rf["time_bwd_max_0.5_lssti"]
        features["time_bwd_max_0.2"] = rf["time_bwd_max_0.2_lssti"]

        features["time_bwd_max_0.5_ratio_red"] = rf["time_bwd_max_0.5_lssty"] / (
            rf["time_bwd_max_0.5_lssty"] + rf["time_bwd_max_0.5_lssti"]
        )
        features["time_bwd_max_0.5_ratio_blue"] = rf["time_bwd_max_0.5_lsstg"] / (
            rf["time_bwd_max_0.5_lsstg"] + rf["time_bwd_max_0.5_lssti"]
        )
        features["time_bwd_max_0.2_ratio_red"] = rf["time_bwd_max_0.2_lssty"] / (
            rf["time_bwd_max_0.2_lssty"] + rf["time_bwd_max_0.2_lssti"]
        )
        features["time_bwd_max_0.2_ratio_blue"] = rf["time_bwd_max_0.2_lsstg"] / (
            rf["time_bwd_max_0.2_lsstg"] + rf["time_bwd_max_0.2_lssti"]
        )

        features["frac_s2n_5"] = rf["count_s2n_5"] / rf["count"]
        features["frac_s2n_-5"] = rf["count_s2n_-5"] / rf["count"]
        features["frac_background"] = rf["frac_background"]

        features["time_width_s2n_5"] = rf["time_width_s2n_5"]

        features["count_max_center"] = rf["count_max_center"]
        features["count_max_rise_20"] = (
            rf["count_max_rise_20"] + features["count_max_center"]
        )
        features["count_max_rise_50"] = (
            rf["count_max_rise_50"] + features["count_max_rise_20"]
        )
        features["count_max_rise_100"] = (
            rf["count_max_rise_100"] + features["count_max_rise_50"]
        )
        features["count_max_fall_20"] = (
            rf["count_max_fall_20"] + features["count_max_center"]
        )
        features["count_max_fall_50"] = (
            rf["count_max_fall_50"] + features["count_max_fall_20"]
        )
        features["count_max_fall_100"] = (
            rf["count_max_fall_100"] + features["count_max_fall_50"]
        )

        all_peak_pos_frac_2 = [
            rf["peaks_pos_lsstu_frac_2"],
            rf["peaks_pos_lsstg_frac_2"],
            rf["peaks_pos_lsstr_frac_2"],
            rf["peaks_pos_lssti_frac_2"],
            rf["peaks_pos_lsstz_frac_2"],
            rf["peaks_pos_lssty_frac_2"],
        ]

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            features["peak_frac_2"] = np.nanmedian(all_peak_pos_frac_2, axis=0)

        features["total_s2n"] = np.sqrt(
            rf["total_s2n_lsstu"] ** 2
            + rf["total_s2n_lsstg"] ** 2
            + rf["total_s2n_lsstr"] ** 2
            + rf["total_s2n_lssti"] ** 2
            + rf["total_s2n_lsstz"] ** 2
            + rf["total_s2n_lssty"] ** 2
        )

        all_frac_percentiles = []
        for percentile in (10, 30, 50, 70, 90):
            frac_percentiles = []
            for band in plasticc_bands:
                percentile_flux = rf["percentile_%s_%d" % (band, percentile)]
                max_flux = rf["max_flux_%s" % band]
                min_flux = rf["min_flux_%s" % band]
                frac_percentiles.append(
                    (percentile_flux - min_flux) / (max_flux - min_flux)
                )
            all_frac_percentiles.append(np.nanmedian(frac_percentiles, axis=0))

        features["percentile_diff_10_50"] = (
            all_frac_percentiles[0] - all_frac_percentiles[2]
        )
        features["percentile_diff_30_50"] = (
            all_frac_percentiles[1] - all_frac_percentiles[2]
        )
        features["percentile_diff_70_50"] = (
            all_frac_percentiles[3] - all_frac_percentiles[2]
        )
        features["percentile_diff_90_50"] = (
            all_frac_percentiles[4] - all_frac_percentiles[2]
        )

        return features


def find_time_to_fractions(fluxes, fractions, forward=True):
    """Find the time for a lightcurve to decline to specific fractions of
    maximum light.

    Parameters
    ----------
    fluxes : numpy.array
        A list of GP-predicted fluxes at 1 day intervals.
    fractions : list
        A decreasing list of the fractions of maximum light that will be found
        (eg: [0.8, 0.5, 0.2]).
    forward : bool
        If True (default), look forward in time. Otherwise, look backward in
        time.

    Returns
    -------
    times : numpy.array
        A list of times for the lightcurve to decline to each of the given
        fractions of maximum light.
    """
    max_time = np.argmax(fluxes)
    max_flux = fluxes[max_time]

    result = np.zeros(len(fractions))
    result[:] = np.nan

    frac_idx = 0

    # Start at maximum light, and move along the spectrum. Whenever we cross
    # one threshold, we add it to the list and keep going. If we hit the end of
    # the array without crossing the threshold, we return a large number for
    # that time.
    offset = 0
    while True:
        offset += 1
        if forward:
            new_time = max_time + offset
            if new_time >= fluxes.shape:
                break
        else:
            new_time = max_time - offset
            if new_time < 0:
                break

        test_flux = fluxes[new_time]
        while test_flux < max_flux * fractions[frac_idx]:
            result[frac_idx] = offset
            frac_idx += 1
            if frac_idx == len(fractions):
                break

        if frac_idx == len(fractions):
            break

    return result


def create_kaggle_predictions(dataset, predictions=None):
    """Add predictions for unknown objects for the Kaggle PLAsTiCC challenge
    using a predefined formula.

    This formula was tuned to the PLAsTiCC dataset, and is not a real method of
    identifying new objects in a dataset.

    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to create predictions for.
    predictions : :class:`pandas.DataFrame` (optional)
        The original predictions for each class. If not specified, the
        dataset's predictions will be used.

    Returns
    -------
    kaggle_predictions : :class:`pandas.DataFrame`
        A pandas DataFrame with the predictions for each class, with class 99
        predictions added.
    """
    if predictions is None:
        predictions = dataset.predictions

    predictions = predictions.copy()

    if 99 in predictions:
        # Remove old 99 prediction and renormalize
        predictions[99] = 0.0
        norm = np.sum(predictions, axis=1)
        predictions = predictions.div(norm, axis=0)

    # Zero out galactic / extragalactic cross predictions.
    galactic_classes = [6, 16, 53, 65, 92]
    for class_name in predictions.columns:
        if class_name in galactic_classes:
            mask = ~dataset.metadata["galactic"]
        else:
            mask = dataset.metadata["galactic"]
        predictions.loc[mask, class_name] = 0.0

    # For extragalactic objects, use a fixed formula. Note: for simplicity we
    # do all objects here, including galactic, and then overwite the galactic
    # ones below.
    predictions[99] = (
        1.0 * predictions[42]
        + 0.6 * predictions[62]
        + 0.2 * predictions[52]
        + 0.2 * predictions[95]
    )

    # For galactic objects, use a flat probability.
    predictions.loc[dataset.metadata["galactic"], 99] = 0.04

    norm = np.sum(predictions, axis=1)
    predictions = predictions.div(norm, axis=0)

    return predictions


def write_kaggle_predictions(dataset, predictions, classifier=None):
    """Write predictions out to disk in the format required for kaggle.

    The class 99 predictions should be added to standard predictions using
    create_kaggle_predictions before calling this method.

    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to write predictions for.
    predictions : :class:`pandas.DataFrame`
        A pandas DataFrame with the predictions for each class, with class 99
        predictions added.
    classifier : str or :class:`Classifier` (optional)
        The classifier to load predictions from. This can be either an instance
        of a :class:`Classifier`, or the name of a classifier. By default, the
        stored classifier is used.
    """
    # Write a CSV instead of the hdf file that is originally saved. We simply
    # change the extension and keep the same path that was originally used.
    hdf_path = dataset.get_predictions_path(classifier)
    csv_path = os.path.splitext(hdf_path)[0] + ".csv"

    # In avocado, we add a prefix to the object IDs. Strip that back off.
    predictions = predictions.copy()
    predictions.index = [int(i.split("_")[1]) for i in predictions.index]
    predictions.index.name = "object_id"

    # Rename the columns to match the Kaggle naming scheme
    predictions.columns = ["class_%d" % i for i in predictions.columns]

    predictions.to_csv(csv_path)
