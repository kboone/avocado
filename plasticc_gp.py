from collections import OrderedDict
import george
from george import kernels
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
import tqdm


# Parameters of the dataset
num_passbands = 6
pad = 100
start_mjd = 59580 - pad
end_mjd = 60675 + pad

# Define class labels, galactic vs extragalactic label and weights
classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1,
                 67: 1, 88: 1, 90: 1, 92: 1, 95: 1, 99: 2}
class_galactic = {6: True, 15: False, 16: True, 42: False, 52: False, 53: True,
                  62: False, 64: False, 65: True, 67: False, 88: False, 90:
                  False, 92: True, 95: False}

# Paths
basedir = '/home/scpdata06/kboone/plasticc'
features_dir = '%s/features' % basedir


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


def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    if y_preds.shape[1] != len(classes):
        # No prediction for 99, pretend that it doesn't exist.
        use_classes = classes[:-1]
    else:
        use_classes = classes

    y_p = y_preds

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[i] for i in use_classes])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)

    return loss


def lgb_multi_weighted_logloss(y_true, y_preds):
    """Wrapper around multi_weighted_logloss that works with lgbm"""
    y_p = y_preds.reshape(y_true.shape[0], len(classes) - 1, order='F')
    loss = multi_weighted_logloss(y_true, y_p)
    return 'wloss', loss, False


def do_predictions(object_ids, features, classifiers):
    pred = 0
    for classifier in classifiers:
        pred += classifier.predict_proba(features) / len(classifiers)

    # Add in flat prediction for class 99. This prediction depends on whether
    # the object is galactic or extragalactic.
    gal_frac_99 = 0.04

    # Weights without 99 included.
    weight_gal = sum([class_weights[class_id] for class_id, is_gal in
                      class_galactic.items() if is_gal])
    weight_extgal = sum([class_weights[class_id] for class_id, is_gal in
                         class_galactic.items() if not is_gal])

    guess_99_gal = gal_frac_99 * class_weights[99] / weight_gal
    guess_99_extgal = (1 - gal_frac_99) * class_weights[99] / weight_extgal

    is_gals = features['hostgal_photoz'] == 0.

    pred_99 = np.array([guess_99_gal if is_gal else guess_99_extgal for is_gal
                        in is_gals])

    stack_pred = np.hstack([pred, pred_99[:, None]])

    # Normalize
    stack_pred = stack_pred / np.sum(stack_pred, axis=1)[:, None]

    # Build a pandas dataframe with the result
    df = pd.DataFrame(index=object_ids, data=stack_pred,
                      columns=['class_%d' % i for i in classes])

    return df


class Dataset(object):
    def __init__(self):
        """Class to represent part of the PLAsTiCC dataset.

        This class can load either the training or validation data, can produce
        features and then can create outputs. The features can also be loaded
        from a file to avoid having to recalculate them every time. Not
        everything has to be loaded at once, but some functions might not work
        if that is the case. I haven't put in the effort to make everything
        safe with regards to random calls, so if something breaks you probably
        just need to load the data that it needs.
        """
        self.flux_data = None
        self.meta_data = None
        self.features = None
        self.dataset_name = None

        # Update this whenever the feature calculation code is updated.
        self._features_version = 1

    def load_training_data(self):
        """Load the training dataset."""
        self.flux_data = pd.read_csv('../data/training_set.csv')
        self.meta_data = pd.read_csv('../data/training_set_metadata.csv')

        self.dataset_name = 'train'

    def load_chunk(self, chunk_idx, load_flux_data=True):
        """Load a chunk from the test dataset.

        I previously split up the dataset into smaller files that can be read
        into memory.

        By default, the flux data is loaded which takes a long time. That can
        be turned off if desired.
        """
        path = ("/home/scpdata06/kboone/plasticc/data_split/"
                "plasticc_split_%04d.h5" % chunk_idx)
        if load_flux_data:
            self.flux_data = pd.read_hdf(path, 'df')
        self.meta_data = pd.read_hdf(path, 'meta')

        self.dataset_name = 'test_%04d' % chunk_idx

    def load_features(self):
        """Load the features for a dataset. This assumes that the features have
        already been created.
        """
        features_path = '%s/features_v%d_%s.h5' % (
            features_dir, self._features_version, self.dataset_name)
        self.raw_features = pd.read_hdf(features_path)

        # Drop keys that we don't want to use in the prediction
        drop_keys = ['object_id', 'hostgal_specz']
        self.features = self.raw_features.drop(drop_keys, 1)

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

        # Add the object id. This shouldn't be used for training a model, but
        # is necessary to identify which row is which when we split things up.
        features['object_id'] = meta['object_id']

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

        return list(features.keys()), np.array(list(features.values()))

    def extract_all_features(self, version=1):
        filename = './train_features_v%d.pkl' % version
        if os.path.exists(filename):
            feature_labels, all_features = pickle.load(open(filename, 'rb'))
        else:
            all_features = []
            for i in tqdm.tqdm(range(len(self.meta_data))):
                feature_labels, features = self.extract_features(i)
                all_features.append(features)
            all_features = np.array(all_features)
            with open(filename, 'wb') as outfile:
                pickle.dump([feature_labels, all_features], outfile)

        return feature_labels, all_features

    def train_classifiers(self):
        """Based off of olivier's kernel"""
        features = self.features
        y = self.meta_data['target']

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        classifiers = []
        importances = pd.DataFrame()
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 14,
            'metric': 'multi_logloss',
            'learning_rate': 0.03,
            'subsample': .9,
            'colsample_bytree': .7,
            'reg_alpha': .01,
            'reg_lambda': .01,
            'min_split_gain': 0.01,
            'min_child_weight': 10,
            'n_estimators': 1000,
            'silent': -1,
            'verbose': -1,
            'max_depth': 3
        }

        # Compute training weights.
        w = y.value_counts()
        norm_class_weights = {i: class_weights[i] * np.sum(w) / w[i] for i in
                              w.index}

        oof_preds = np.zeros((len(features), np.unique(y).shape[0]))
        for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
            trn_x, trn_y = features.iloc[trn_], y.iloc[trn_]
            val_x, val_y = features.iloc[val_], y.iloc[val_]
            trn_weights = trn_y.map(norm_class_weights)
            val_weights = val_y.map(norm_class_weights)

            classifier = lgb.LGBMClassifier(**lgb_params)
            classifier.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric=lgb_multi_weighted_logloss,
                verbose=100,
                early_stopping_rounds=50,
                sample_weight=trn_weights,
                eval_sample_weight=[trn_weights, val_weights],
            )
            val_preds = classifier.predict_proba(
                val_x, num_iteration=classifier.best_iteration_)

            oof_preds[val_, :] = val_preds

            imp_df = pd.DataFrame()
            imp_df['feature'] = features.columns
            imp_df['gain'] = classifier.feature_importances_
            imp_df['fold'] = fold_ + 1
            importances = pd.concat([importances, imp_df], axis=0, sort=False)

            classifiers.append(classifier)

        print('MULTI WEIGHTED LOG LOSS : %.5f ' %
              multi_weighted_logloss(y_true=y, y_preds=oof_preds))

        # Build a pandas dataframe with the out of fold predictions
        oof_preds_df = pd.DataFrame(data=oof_preds,
                                    columns=['class_%d' % i for i in
                                             classes[:-1]])

        # Save results of the fit internally, and return the fitted
        # classifiers.
        self.importances = importances
        self.fit_preds = oof_preds_df

        return classifiers
