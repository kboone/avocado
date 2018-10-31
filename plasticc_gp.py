import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import george
from george import kernels
from george.modeling import Model
from collections import OrderedDict
from scipy.optimize import minimize


# Parameters of the dataset
num_passbands = 6
pad = 100
start_mjd = 59580 - pad
end_mjd = 60675 + pad


def find_time_to_fractions(fluxes, fractions, forward=True):
    """Find the time for a lightcurve to decline to a specific fraction of
    maximum light.

    fractions should be a decreasing list of the fractions of maximum light
    that will be found (eg: [0.8, 0.5, 0.2]).
    """
    max_time = np.argmax(fluxes)
    max_flux = fluxes[max_time]

    result = np.ones(len(fractions)) * 99999

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


class Dataset(object):
    def load_training_data(self):
        """Load the training dataset."""
        self.flux_data = pd.read_csv('../data/training_set.csv')
        self.meta_data = pd.read_csv('../data/training_set_metadata.csv')

    def load_chunk(self, chunk_idx):
        """Load a chunk from the test dataset.

        I previously split up the dataset into smaller files that can be read
        into memory.
        """
        path = '../data_split/plasticc_split_%04d.h5' % chunk_idx
        self.flux_data = pd.read_hdf(path, 'df')
        self.meta_data = pd.read_hdf(path, 'df')

    def get_gp_data(self, idx, target=None, verbose=False):
        if target is not None:
            target_data = self.meta_data[self.meta_data['target'] == target]
            object_meta = target_data.iloc[idx]
        else:
            object_meta = self.meta_data.iloc[idx]

        if verbose:
            print(object_meta)

        object_id = object_meta['object_id']

        object_data = self.flux_data[self.flux_data['object_id'] == object_id]

        times = []
        fluxes = []
        bands = []
        flux_errs = []

        # The zeropoints were arbitrarily set from the first image. Pick the
        # 20th percentile of all observations in each channel as a new
        # zeropoint. This has good performance when there are supernova-like
        # bursts in the image, even if they are quite wide.
        for passband in range(num_passbands):
            band_data = object_data[object_data['passband'] == passband]
            ref_flux = np.percentile(band_data['flux'], 20)

            for idx, row in band_data.iterrows():
                times.append(row['mjd'] - start_mjd)
                fluxes.append(row['flux'] - ref_flux)
                bands.append(passband)
                flux_errs.append(row['flux_err'])

        times = np.array(times)
        bands = np.array(bands)
        fluxes = np.array(fluxes)
        flux_errs = np.array(flux_errs)

        scale = np.max(np.abs(fluxes))

        gp_data = {
            'meta': object_meta,
            'times': times,
            'bands': bands,
            'scale': scale,
            'fluxes': fluxes,
            'flux_errs': flux_errs,
        }

        return gp_data

    def fit_gp(self, idx, target=None, verbose=False):
        gp_data = self.get_gp_data(idx, target, verbose)

        # GP kernel. We use a 2-dimensional Matern kernel to model the
        # transient. The kernel amplitude is fixed to a fraction of the maximum
        # value in the data, and the kernel width in the wavelength direction
        # is also fixed. We fit for the kernel width in the time direction as
        # different transients evolve on very different time scales.
        kernel = ((0.2*gp_data['scale'])**2 *
                  kernels.Matern32Kernel([20.**2, 5**2], ndim=2))

        # print(kernel.get_parameter_names())
        kernel.freeze_parameter('k1:log_constant')
        kernel.freeze_parameter('k2:metric:log_M_1_1')

        gp = george.GP(kernel)

        if verbose:
            print(kernel.get_parameter_dict())

        x_data = np.vstack([gp_data['times'], gp_data['bands']]).T

        gp.compute(x_data, gp_data['flux_errs'])

        fluxes = gp_data['fluxes']

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(fluxes)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(fluxes)

        # print(np.exp(gp.get_parameter_vector()))

        fit_result = minimize(
            neg_ln_like,
            gp.get_parameter_vector(),
            jac=grad_neg_ln_like,
            # bounds=[(-30, 30), (0, 10), (0, 5)],
            # bounds=[(0, 10), (0, 5)],
            bounds=[(0, np.log(1000**2))],
            # options={'ftol': 1e-5}
        )

        if not fit_result.success:
            print("Fit failed for %d!" % idx)

        # print(-gp.log_likelihood(fluxes))
        # print(np.exp(fit_result.x))

        gp.set_parameter_vector(fit_result.x)

        if verbose:
            print(fit_result)
            print(kernel.get_parameter_dict())

        pred = []
        pred_times = np.arange(end_mjd - start_mjd + 1)

        for band in range(6):
            pred_bands = np.ones(len(pred_times)) * band
            pred_x_data = np.vstack([pred_times, pred_bands]).T
            # pred, pred_var = gp.predict(fluxes, pred_x_data, return_var=True)
            # band_pred, pred_var = gp.predict(fluxes, pred_x_data,
            # return_var=True)
            # band_pred = gp.predict(fluxes, pred_x_data, return_var=False)
            band_pred = gp.predict(fluxes, pred_x_data, return_cov=False)
            pred.append(band_pred)
        pred = np.array(pred)

        # Add results of the GP fit to the gp_data dictionary.
        gp_data['pred_times'] = pred_times
        gp_data['pred'] = pred
        gp_data['fit_parameters'] = fit_result.x

        return gp_data

    def plot_gp(self, idx, target=None, verbose=False):
        result = self.fit_gp(idx, target, verbose)

        plt.figure()

        for band in range(num_passbands):
            cut = result['bands'] == band
            color = 'C%d' % band
            plt.errorbar(result['times'][cut], result['fluxes'][cut],
                         result['flux_errs'][cut], fmt='o', c=color)
            plt.plot(result['pred_times'], result['pred'][band], c=color,
                     label=band)

        plt.legend()

    def plot_gp_interactive(self):
        """Make an interactive plot of the GP output.

        This requires the ipywidgets package to be set up, and has only been
        tested in jupyter-lab.
        """
        from ipywidgets import interact, IntSlider, Dropdown

        targets = np.unique(self.meta_data['target'])

        idx_widget = IntSlider(min=0, max=1)
        target_widget = Dropdown(options=targets, index=0)

        def update_idx_range(*args):
            idx_widget.max = np.sum(self.meta_data['target'] ==
                                    target_widget.value) - 1
            target_widget.observe(update_idx_range, 'value')
        update_idx_range()

        interact(self.plot_gp, target=target_widget, idx=idx_widget)

    def extract_features(self, idx, target=None):
        """Extract features from a target"""
        features = OrderedDict()

        # Fit the GP and produce an output model
        gp_data = self.fit_gp(idx, target)
        times = gp_data['times']
        fluxes = gp_data['fluxes']
        flux_errs = gp_data['flux_errs']
        s2ns = fluxes / flux_errs
        pred = gp_data['pred']
        meta = gp_data['meta']

        # Features from the meta data
        features['hostgal_specz'] = meta['hostgal_specz']
        features['hostgal_photoz'] = meta['hostgal_photoz']
        features['hostgal_photoz_err'] = meta['hostgal_photoz_err']
        features['ra'] = meta['ra']
        features['decl'] = meta['decl']
        features['gal_l'] = meta['gal_l']
        features['gal_b'] = meta['gal_b']
        features['distmod'] = meta['distmod']
        features['mwebv'] = meta['mwebv']
        features['ddf'] = meta['ddf']

        # Features from GP fit parameters
        for i, fit_parameter in enumerate(gp_data['fit_parameters']):
            features['gp_fit_%d' % i] = fit_parameter

        # Maximum fluxes and times.
        max_times = np.argmax(pred, axis=1)
        med_max_time = np.median(max_times)
        max_dts = max_times - med_max_time
        max_fluxes = [pred[band, time] for band, time in enumerate(max_times)]
        for band, (max_flux, max_dt) in enumerate(zip(max_fluxes, max_dts)):
            features['max_flux_%d' % band] = max_flux
            features['max_dt_%d' % band] = max_dt

        # Find times to fractions of the peak amplitude
        fractions = [0.8, 0.5, 0.2]
        for band in range(num_passbands):
            forward_times = find_time_to_fractions(pred[band], fractions)
            backward_times = find_time_to_fractions(pred[band], fractions,
                                                    forward=False)
            for fraction, forward_time, backward_time in \
                    zip(fractions, forward_times, backward_times):
                features['frac_time_fwd_%.1f_%d' % (fraction, band)] = \
                    forward_time
                features['frac_time_bwd_%.1f_%d' % (fraction, band)] = \
                    backward_time

        # Count the number of data points with significant positive/negative
        # fluxes
        thresholds = [-10, -5, -3, 3, 5, 10]
        for threshold in thresholds:
            if threshold < 0:
                count = np.sum(s2ns < threshold)
            else:
                count = np.sum(s2ns > threshold)
            features['count_s2n_%d' % threshold] = count

        # Count the fraction of data points that are "background", i.e. less
        # than a 3 sigma detection of something.
        features['frac_background'] = np.sum(np.abs(s2ns) < 3) / len(s2ns)

        # Count the time delay between the first and last significant fluxes
        thresholds = [5, 10, 20]
        for threshold in thresholds:
            significant_times = times[np.abs(s2ns) > threshold]
            if len(significant_times) < 2:
                dt = -1
            else:
                dt = np.max(significant_times) - np.min(significant_times)
            features['time_width_s2n_%d' % threshold] = dt

        # Count how many data points are within a certain number of days of
        # maximum light. This provides some estimate of the robustness of the
        # determination of maximum light and rise/fall times.
        time_bins = [
            (-5, 5, 'center'),
            (-20, -5, 'rise_20'),
            (-50, -20, 'rise_50'),
            (-200, -50, 'rise_200'),
            (5, 20, 'fall_20'),
            (20, 50, 'fall_50'),
            (50, 200, 'fall_200'),
        ]
        for start, end, label in time_bins:
            diff_times = times - med_max_time
            count = np.sum((diff_times > start) & (diff_times < end))
            features['count_max_%s' % label] = count

        # Count 

        return list(features.keys()), np.array(list(features.values()))
