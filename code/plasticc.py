from collections import OrderedDict, defaultdict
import george
from george import kernels
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from scipy.signal import find_peaks
from scipy.special import erf
from functools import partial
import tqdm

from settings import settings

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

# Band names
band_names = ['u', 'g', 'r', 'i', 'z', 'y']
band_wavelengths = [3671., 4827., 6223., 7546., 8691., 9710.]
band_colors = ['C6', 'C4', 'C0', 'C2', 'C3', 'goldenrod']

# Reverse engineered cosmology used in sims
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)


class PlasticcException(Exception):
    pass


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


def do_predictions_flatprob(object_ids, features, classifiers):
    pred = 0
    for classifier in classifiers:
        pred += (
            classifier.predict_proba(
                features, num_iteration=classifier.best_iteration_)
        ) / len(classifiers)

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


def do_predictions(object_ids, features, classifiers, gal_outlier_score=0.25,
                   extgal_outlier_score=1.4):
    print("OLD!!! DON'T USE!")
    is_gal = features['hostgal_photoz'] == 0.
    base_class_99_scores = np.zeros((len(features), 1))
    base_class_99_scores[is_gal] = gal_outlier_score
    base_class_99_scores[~is_gal] = extgal_outlier_score

    pred = 0
    for classifier in classifiers:
        # Get base scores
        raw_scores = classifier.predict_proba(
            features, raw_score=True, num_iteration=classifier.best_iteration_
        )
        max_scores = np.max(raw_scores, axis=1)[:, None]
        class_99_scores = np.clip(base_class_99_scores, None,
                                  max_scores)

        # Add in class 99 scores.
        scores = np.hstack([raw_scores, class_99_scores])

        # Turn the scores into a prediction
        iter_pred = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, None]

        pred += iter_pred / len(classifiers)

    # Build a pandas dataframe with the result
    df = pd.DataFrame(index=object_ids, data=pred,
                      columns=['class_%d' % i for i in classes])

    return df


def do_scores(object_ids, features, classifiers):
    scores = []
    for classifier in classifiers:
        scores.append(classifier.predict_proba(
            features, raw_score=True,
            num_iteration=classifier.best_iteration_))
    scores = np.array(scores)

    return scores


def convert_scores(meta, scores, gal_outlier_score=0.4,
                   extgal_outlier_score=1.4):
    is_gal = meta['hostgal_photoz'] == 0.
    base_class_99_scores = np.zeros((len(meta), 1))
    base_class_99_scores[is_gal] = gal_outlier_score
    base_class_99_scores[~is_gal] = extgal_outlier_score

    pred = 0
    for iter_scores in scores:
        # Iterate over each classifier's scores if there were more than one.
        # Get base scores
        # max_scores = np.max(iter_scores, axis=1)[:, None]
        max_scores = np.percentile(iter_scores, 100 * 12.5/13, axis=1)[:, None]
        class_99_scores = np.clip(base_class_99_scores, None, max_scores)

        # Add in class 99 scores.
        iter_full_scores = np.hstack([iter_scores, class_99_scores])

        # Turn the scores into a prediction
        iter_pred = np.exp(iter_full_scores) / np.sum(np.exp(iter_full_scores),
                                                      axis=1)[:, None]

        pred += iter_pred / len(scores)

    print("Mean gal 99: %.5f" % np.mean(pred[is_gal, -1]))
    print("Mean ext 99: %.5f" % np.mean(pred[~is_gal, -1]))

    # Build a pandas dataframe with the result
    df = pd.DataFrame(index=meta['object_id'], data=pred,
                      columns=['class_%d' % i for i in classes])

    return df


def convert_scores_2(meta, scores, s2n, gal_outlier_score=-2.,
                     extgal_outlier_score=-0.8):
    is_gal = meta['hostgal_photoz'] == 0.
    base_class_99_scores = np.zeros((len(meta), 1))
    base_class_99_scores[is_gal] = gal_outlier_score
    base_class_99_scores[~is_gal] = extgal_outlier_score
    base_class_99_scores[:, 0] += 1.5*np.log10(s2n)

    pred = 0
    for iter_scores in scores:
        # Iterate over each classifier's scores if there were more than one.
        # Get base scores
        # max_scores = np.max(iter_scores, axis=1)[:, None]
        max_scores = np.percentile(iter_scores, 100 * 12.5/13, axis=1)[:, None]
        class_99_scores = np.clip(base_class_99_scores, None, max_scores)

        # Add in class 99 scores.
        iter_full_scores = np.hstack([iter_scores, class_99_scores])

        # Turn the scores into a prediction
        iter_pred = np.exp(iter_full_scores) / np.sum(np.exp(iter_full_scores),
                                                      axis=1)[:, None]

        pred += iter_pred / len(scores)

    print("Mean gal 99: %.5f" % np.mean(pred[is_gal, -1]))
    print("Mean ext 99: %.5f" % np.mean(pred[~is_gal, -1]))

    # Build a pandas dataframe with the result
    df = pd.DataFrame(index=meta['object_id'], data=pred,
                      columns=['class_%d' % i for i in classes])

    return df


def fit_classifier(train_x, train_y, train_weights, eval_x=None, eval_y=None,
                   eval_weights=None, **kwargs):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 14,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        # 'bagging_fraction': .75,
        # 'bagging_freq': 5,
        'colsample_bytree': .5,
        'reg_alpha': 0.,
        'reg_lambda': 0.,
        'min_split_gain': 10.,
        'min_child_weight': 2000.,
        'n_estimators': 5000,
        'silent': -1,
        'verbose': -1,
        'max_depth': 7,
        'num_leaves': 50,
    }

    lgb_params.update(kwargs)

    fit_params = {
        'verbose': 100,
        'sample_weight': train_weights,
    }

    if eval_x is not None:
        fit_params['eval_set'] = [(eval_x, eval_y)]
        fit_params['eval_metric'] = lgb_multi_weighted_logloss
        fit_params['early_stopping_rounds'] = 50
        fit_params['eval_sample_weight'] = [eval_weights]

    classifier = lgb.LGBMClassifier(**lgb_params)
    classifier.fit(train_x, train_y, **fit_params)

    return classifier


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
        self.gp_fits = None
        self.dataset_name = None

        # Update this whenever the feature calculation code is updated.
        self._features_version = settings['FEATURES_VERSION']

        # Update this whenever the augmentation code is updated.
        self._augment_version = settings['AUGMENT_VERSION']

    def load_training_data(self):
        """Load the training dataset."""
        self.flux_data = pd.read_csv(settings['RAW_TRAINING_PATH'])
        self.meta_data = pd.read_csv(settings["RAW_TRAINING_METADATA_PATH"])

        # Label folds
        y = self.meta_data['target']
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        kfold_indices = -1*np.ones(len(y))
        for idx, (fold_train, fold_val) in enumerate(folds.split(y, y)):
            kfold_indices[fold_val] = idx
        self.meta_data['fold'] = kfold_indices

        self.dataset_name = 'train'

    def load_test_data(self, unblind=False):
        """Load the test metadata."""
        if unblind:
            meta_path = settings["UNBLIND_TEST_METADATA_PATH"]
        else:
            meta_path = settings["RAW_TEST_METADATA_PATH"]

        self.meta_data = pd.read_csv(meta_path)
        self.dataset_name = 'test'

    def load_chunk(self, chunk_idx, load_flux_data=True):
        """Load a chunk from the test dataset.

        I previously split up the dataset into smaller files that can be read
        into memory.

        By default, the flux data is loaded which takes a long time. That can
        be turned off if desired.
        """
        path = settings["SPLIT_TEST_PATH_FORMAT"] % chunk_idx
        if load_flux_data:
            self.flux_data = pd.read_hdf(path, 'df')
        self.meta_data = pd.read_hdf(path, 'meta')

        self.dataset_name = 'test_%04d' % chunk_idx

    def load_augment(self, num_augments=None, base_name='train', chunk=None,
                     merge=False):
        """Load an augmented dataset."""
        if num_augments is None:
            num_augments = settings['NUM_AUGMENTS']

        if merge:
            dataset_name = '%s_augment_v%d_%d_merge' % (
                base_name, self._augment_version, num_augments)
        elif chunk is not None:
            dataset_name = '%s_augment_v%d_%d_%03d' % (
                base_name, self._augment_version, num_augments, chunk)
        else:
            dataset_name = '%s_augment_v%d_%d' % (
                base_name, self._augment_version, num_augments)
        path = '%s/%s.h5' % (settings['AUGMENT_DIR'], dataset_name)

        self.flux_data = pd.read_hdf(path, 'df')
        self.meta_data = pd.read_hdf(path, 'meta')

        self.dataset_name = dataset_name

    def load_unblind_meta(self):
        """Load the unblinded meta data."""
        meta_path = settings["UNBLIND_TEST_METADATA_PATH"]

        unblind_meta = pd.read_csv(meta_path)

        self.meta_data = self.meta_data.merge(
            unblind_meta,
            how='left',
            on='object_id',
            suffixes=('', '_unblind'),
        )

    def load_predictions(self, num_augments=None):
        """Load the predictions for some part of the test set"""
        prediction_path = settings["SUBMISSIONS_PROBE99_PATH_FORMAT"]

        if num_augments is None:
            num_augments = settings['NUM_AUGMENTS']

        predictions = pd.read_csv(prediction_path % num_augments)

        self.meta_data = self.meta_data.merge(
            predictions,
            how='left',
            on='object_id',
            suffixes=('', '_prediction'),
        )

    def load_gp_fits(self):
        """Load all of the Gaussian process fits for a dataset"""
        with open(self.gp_fits_path, 'rb') as fin:
            gp_fits = pickle.load(fin)
        self.gp_fits = gp_fits

    @property
    def features_path(self):
        """Path to the features file for this dataset"""
        features_path = settings['FEATURES_PATH_FORMAT'] % (
            self._features_version, self.dataset_name)
        return features_path

    @property
    def gp_fits_path(self):
        """Path to the Gaussian process fits file for this dataset"""
        gp_path = settings['GP_FITS_PATH_FORMAT'] % (
            self._features_version, self.dataset_name)
        return gp_path

    @property
    def salt_fits_path(self):
        """Path to the features file for this dataset"""
        salt_fits_path = settings['SALT_FITS_PATH_FORMAT'] % (
            self.dataset_name)
        return salt_fits_path

    def load_simple_features(self, apply_photoz=False):
        """Load the features for a dataset and postprocess them.

        This assumes that the features have already been created.
        """
        self.raw_features = pd.read_hdf(self.features_path)

        rf = self.raw_features

        max_flux = rf['max_flux_3'] - rf['min_flux_3']
        # max_flux = np.clip(rf['max_flux_3'], 0.001, None)
        max_mag = -2.5*np.log10(np.abs(max_flux))

        # If desired, undo the photoz
        if apply_photoz:
            distmod = rf['distmod']
            photoz = rf['hostgal_photoz']

            # Turn max brightness into luminosity for extragalactic objects
            extgal_mask = photoz != 0
            max_mag[extgal_mask] = (
                max_mag[extgal_mask] - distmod[extgal_mask] + 28
            )

            time_scale = (1 + rf['hostgal_photoz'])
        else:
            time_scale = 1

        # Keys that we want to use in the prediction.
        use_keys = [
            'hostgal_photoz',
            'hostgal_photoz_err',
            # 'count',
        ]

        features = rf[use_keys].copy()

        features['length_scale'] = rf['gp_fit_1'] / time_scale

        # features['max_flux'] = rf['max_flux_3']
        features['max_mag'] = max_mag

        features['pos_flux_ratio'] = rf['max_flux_3'] / max_flux
        features['max_flux_ratio_r'] = (
            (rf['max_flux_5'] - rf['max_flux_3']) /
            (np.abs(rf['max_flux_5']) + np.abs(rf['max_flux_3']))
        )
        features['max_flux_ratio_b'] = (
            (rf['max_flux_3'] - rf['max_flux_1']) /
            (np.abs(rf['max_flux_3']) + np.abs(rf['max_flux_1']))
        )

        # features['min_flux'] = rf['min_flux_3']
        features['min_flux_ratio_r'] = (
            (rf['min_flux_5'] - rf['min_flux_3']) /
            (np.abs(rf['min_flux_5']) + np.abs(rf['min_flux_3']))
        )
        features['min_flux_ratio_b'] = (
            (rf['min_flux_3'] - rf['min_flux_1']) /
            (np.abs(rf['min_flux_3']) + np.abs(rf['min_flux_1']))
        )

        features['max_dt'] = (rf['max_dt_5'] - rf['max_dt_1']) / time_scale

        features['positive_width'] = rf['positive_width_3'] / time_scale
        features['negative_width'] = rf['negative_width_3'] / time_scale

        features['frac_time_fwd_0.8'] = rf['frac_time_fwd_0.8_3'] / time_scale
        features['frac_time_fwd_0.5'] = rf['frac_time_fwd_0.5_3'] / time_scale
        features['frac_time_fwd_0.2'] = rf['frac_time_fwd_0.2_3'] / time_scale

        features['ratio_r_time_fwd_0.8'] = (
            rf['frac_time_fwd_0.8_3'] / rf['frac_time_fwd_0.8_5'])
        features['ratio_b_time_fwd_0.8'] = (
            rf['frac_time_fwd_0.8_3'] / rf['frac_time_fwd_0.8_1'])
        features['ratio_r_time_fwd_0.5'] = (
            rf['frac_time_fwd_0.5_3'] / rf['frac_time_fwd_0.5_5'])
        features['ratio_b_time_fwd_0.5'] = (
            rf['frac_time_fwd_0.5_3'] / rf['frac_time_fwd_0.5_1'])
        features['ratio_r_time_fwd_0.2'] = (
            rf['frac_time_fwd_0.2_3'] / rf['frac_time_fwd_0.2_5'])
        features['ratio_b_time_fwd_0.5'] = (
            rf['frac_time_fwd_0.2_3'] / rf['frac_time_fwd_0.2_1'])

        features['frac_time_bwd_0.8'] = rf['frac_time_bwd_0.8_3'] / time_scale
        features['frac_time_bwd_0.5'] = rf['frac_time_bwd_0.5_3'] / time_scale
        features['frac_time_bwd_0.2'] = rf['frac_time_bwd_0.2_3'] / time_scale

        features['ratio_r_time_bwd_0.8'] = (
            rf['frac_time_bwd_0.8_3'] / rf['frac_time_bwd_0.8_5'])
        features['ratio_b_time_bwd_0.8'] = (
            rf['frac_time_bwd_0.8_3'] / rf['frac_time_bwd_0.8_1'])
        features['ratio_r_time_bwd_0.5'] = (
            rf['frac_time_bwd_0.5_3'] / rf['frac_time_bwd_0.5_5'])
        features['ratio_b_time_bwd_0.5'] = (
            rf['frac_time_bwd_0.5_3'] / rf['frac_time_bwd_0.5_1'])
        features['ratio_r_time_bwd_0.2'] = (
            rf['frac_time_bwd_0.2_3'] / rf['frac_time_bwd_0.2_5'])
        features['ratio_b_time_bwd_0.5'] = (
            rf['frac_time_bwd_0.2_3'] / rf['frac_time_bwd_0.2_1'])

        features['frac_s2n_5'] = rf['count_s2n_5'] / rf['count']
        features['frac_s2n_-5'] = rf['count_s2n_-5'] / rf['count']
        features['frac_background'] = rf['frac_background']

        features['time_width_s2n_5'] = rf['time_width_s2n_5'] / time_scale

        features['count_max_center'] = rf['count_max_center']
        features['count_max_rise_20'] = rf['count_max_rise_20']
        features['count_max_rise_50'] = rf['count_max_rise_50']
        features['count_max_rise_100'] = rf['count_max_rise_100']
        features['count_max_fall_20'] = rf['count_max_fall_20']
        features['count_max_fall_50'] = rf['count_max_fall_50']
        features['count_max_fall_100'] = rf['count_max_fall_100']

        features['num_peaks'] = np.nanmedian([
            rf['peaks_pos_0_count'],
            rf['peaks_pos_1_count'],
            rf['peaks_pos_2_count'],
            rf['peaks_pos_3_count'],
            rf['peaks_pos_4_count'],
            rf['peaks_pos_5_count']
        ], axis=0)

        features['peak_frac_2'] = np.nanmedian([
            rf['peaks_pos_0_frac_2'],
            rf['peaks_pos_1_frac_2'],
            rf['peaks_pos_2_frac_2'],
            rf['peaks_pos_3_frac_2'],
            rf['peaks_pos_4_frac_2'],
            rf['peaks_pos_5_frac_2']
        ], axis=0)

        features['peak_frac_3'] = np.nanmedian([
            rf['peaks_pos_0_frac_3'],
            rf['peaks_pos_1_frac_3'],
            rf['peaks_pos_2_frac_3'],
            rf['peaks_pos_3_frac_3'],
            rf['peaks_pos_4_frac_3'],
            rf['peaks_pos_5_frac_3']
        ], axis=0)

        features['total_s2n'] = (
            rf['total_s2n_0'] +
            rf['total_s2n_1'] +
            rf['total_s2n_2'] +
            rf['total_s2n_3'] +
            rf['total_s2n_4'] +
            rf['total_s2n_5']
        )

        self.features = features

    def load_features(self):
        """Load the features for a dataset and postprocess them.

        This assumes that the features have already been created.
        """
        self.raw_features = pd.read_hdf(self.features_path)

        # Drop keys that we don't want to use in the prediction
        drop_keys = [
            'object_id',
            'hostgal_specz',
            # 'hostgal_photoz',
            # 'distmod',
            'ra',
            'decl',
            'gal_l',
            'gal_b',
            'mwebv',
            'ddf',
            'max_time',
            # 'hostgal_photoz',
        ]

        features = self.raw_features
        for key in drop_keys:
            try:
                features = features.drop(key, 1)
            except KeyError:
                # Key doesn't exist in this version. Ignore it.
                pass

        self.features = features

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

            # ref_flux = np.percentile(band_data['flux'], 20)
            # ref_flux = np.median(band_data['flux'])

            # Determine the reference flux level for this band from the final
            # observations.
            ref_count = 7
            ref_indices = np.argsort(band_data['mjd'])[-ref_count:]
            ref_flux = min(0, np.median(band_data['flux'].iloc[ref_indices]))

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

    def get_gp_data(self, idx=None, target=None, object_id=None, verbose=False,
                    fix_background=True):
        if object_id is not None:
            # Get the object id directly
            object_meta = self.meta_data[self.meta_data['object_id'] ==
                                         object_id].iloc[0]
        elif target is not None:
            target_data = self.meta_data[self.meta_data['target'] == target]
            object_meta = target_data.iloc[idx]
        else:
            object_meta = self.meta_data.iloc[idx]

        if verbose:
            print(object_meta)

        object_id = object_meta['object_id']
        object_data = self.flux_data[self.flux_data['object_id'] == object_id]

        return self._get_gp_data(object_meta, object_data, fix_background)

    def fit_gp(self, idx=None, target=None, object_id=None, object_meta=None,
               object_data=None, verbose=False, guess_length_scale=20.,
               fix_scale=False, fix_background=True):
        if idx is not None or object_id is not None:
            # idx or object_id was specified, pull from the internal data
            gp_data = self.get_gp_data(idx, target, object_id, verbose,
                                       fix_background)
        else:
            # The meta data and flux data can also be directly specified.
            gp_data = self._get_gp_data(object_meta, object_data,
                                        fix_background)

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

    def fit_all_gps(self):
        """Do Gaussian process fits for all objects in the dataset and save
        them to a file.
        """
        # Do all of the GP fits
        gp_fits = []
        print("Fitting all GPs")
        sys.stdout.flush()
        for i in tqdm.tqdm(range(len(self.meta_data))):
            gp, gp_data = self.fit_gp(i)
            gp_fits.append(gp)

        # Save them to disk
        with open(self.gp_fits_path, 'wb') as fout:
            pickle.dump(gp_fits, fout)

        self.gp_fits = gp_fits

    def predict_gp(self, *args, uncertainties=False, **kwargs):
        gp, gp_data = self.fit_gp(*args, **kwargs)

        pred = []
        pred_var = []
        pred_times = np.arange(end_mjd - start_mjd + 1)

        for band in range(6):
            pred_wavelengths = (
                np.ones(len(pred_times)) * band_wavelengths[band]
            )
            pred_x_data = np.vstack([pred_times, pred_wavelengths]).T
            if uncertainties:
                band_pred, band_pred_var = gp(pred_x_data, return_var=True)
                pred_var.append(band_pred_var)
            else:
                band_pred = gp(pred_x_data, return_cov=False)
            pred.append(band_pred)
        pred = np.array(pred)

        if uncertainties:
            pred_var = np.array(pred_var)
            gp_data['pred_var'] = pred_var

        # Add results of the GP prediction to the gp_data dictionary.
        gp_data['pred_times'] = pred_times
        gp_data['pred'] = pred

        return gp_data

    def plot_gp(self, *args, data_only=False, **kwargs):
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

    def plot_gp_interactive(self):
        """Make an interactive plot of the GP output.

        This requires the ipywidgets package to be set up, and has only been
        tested in jupyter-lab.
        """
        from ipywidgets import interact, IntSlider, Dropdown, fixed

        targets = np.unique(self.meta_data['target'])

        idx_widget = IntSlider(min=0, max=1)
        target_widget = Dropdown(options=targets, index=0)

        def update_idx_range(*args):
            idx_widget.max = np.sum(self.meta_data['target'] ==
                                    target_widget.value) - 1
            target_widget.observe(update_idx_range, 'value')
        update_idx_range()

        interact(self.plot_gp, idx=idx_widget, target=target_widget,
                 object_meta=fixed(None), object_data=fixed(None),
                 uncertainties=True, data_only=False, verbose=False,
                 fix_background=True)

    def extract_features(self, *args, **kwargs):
        """Extract features from a target"""
        features = OrderedDict()

        # Fit the GP and produce an output model
        gp_data = self.predict_gp(*args, **kwargs)
        times = gp_data['times']
        fluxes = gp_data['fluxes']
        flux_errs = gp_data['flux_errs']
        bands = gp_data['bands']
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

        # Count how many observations there are
        features['count'] = len(fluxes)

        # Features from GP fit parameters
        for i, fit_parameter in enumerate(gp_data['fit_parameters']):
            features['gp_fit_%d' % i] = fit_parameter

        # Maximum fluxes and times.
        max_times = np.argmax(pred, axis=1)
        med_max_time = np.median(max_times)
        max_dts = max_times - med_max_time
        max_fluxes = np.array([pred[band, time] for band, time in
                               enumerate(max_times)])
        features['max_time'] = med_max_time
        for band, (max_flux, max_dt) in enumerate(zip(max_fluxes, max_dts)):
            features['max_flux_%d' % band] = max_flux
            features['max_dt_%d' % band] = max_dt

        # Minimum fluxes.
        min_fluxes = np.min(pred, axis=1)
        for band, min_flux in enumerate(min_fluxes):
            features['min_flux_%d' % band] = min_flux

        # Calculate the positive and negative integrals of the lightcurve,
        # normalized to the respective peak fluxes. This gives a measure of the
        # "width" of the lightcurve, even for non-bursty objects.
        positive_widths = np.sum(np.clip(pred, 0, None), axis=1) / max_fluxes
        negative_widths = np.sum(np.clip(pred, None, 0), axis=1) / min_fluxes
        for band in range(num_passbands):
            features['positive_width_%d' % band] = positive_widths[band]
            features['negative_width_%d' % band] = negative_widths[band]

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
        thresholds = [-20, -10, -5, -3, 3, 5, 10, 20]
        for threshold in thresholds:
            if threshold < 0:
                count = np.sum(s2ns < threshold)
            else:
                count = np.sum(s2ns > threshold)
            features['count_s2n_%d' % threshold] = count

        # Count the fraction of data points that are "background", i.e. less
        # than a 3 sigma detection of something.
        features['frac_background'] = np.sum(np.abs(s2ns) < 3) / len(s2ns)

        # Sum up the total signal-to-noise in each band
        for band in range(6):
            mask = bands == band
            band_fluxes = fluxes[mask]
            band_flux_errs = flux_errs[mask]
            total_band_s2n = np.sqrt(np.sum((band_fluxes / band_flux_errs)**2))
            features['total_s2n_%d' % band] = total_band_s2n

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
            (-100, -50, 'rise_100'),
            (-200, -100, 'rise_200'),
            (-300, -200, 'rise_300'),
            (-400, -300, 'rise_400'),
            (-500, -400, 'rise_500'),
            (-600, -500, 'rise_600'),
            (-700, -600, 'rise_700'),
            (-800, -700, 'rise_800'),
            (5, 20, 'fall_20'),
            (20, 50, 'fall_50'),
            (50, 100, 'fall_100'),
            (100, 200, 'fall_200'),
            (200, 300, 'fall_300'),
            (300, 400, 'fall_400'),
            (400, 500, 'fall_500'),
            (500, 600, 'fall_600'),
            (600, 700, 'fall_700'),
            (700, 800, 'fall_800'),
        ]
        for start, end, label in time_bins:
            diff_times = times - med_max_time

            mask = (diff_times > start) & (diff_times < end)

            # Count how many observations there are in the time bin
            count = np.sum(mask)
            features['count_max_%s' % label] = count

            # Measure the GP flux level relative to the peak flux. We do this
            # by taking the median flux in each band and comparing it to the
            # peak flux.
            bin_start = np.clip(int(med_max_time + start), 0, None)
            bin_end = np.clip(int(med_max_time + end), 0, None)

            if bin_start == bin_end:
                scale_pred = np.nan
                bin_mean_fluxes = np.nan
                bin_std_fluxes = np.nan
            else:
                scale_pred = pred[:, bin_start:bin_end] / max_fluxes[:, None]
                bin_mean_fluxes = np.mean(scale_pred)
                bin_std_fluxes = np.std(scale_pred)
            features['mean_%s' % label] = bin_mean_fluxes
            features['std_%s' % label] = bin_std_fluxes

        # Do peak detection on the GP output
        for positive in (True, False):
            for band in range(num_passbands):
                if positive:
                    band_flux = pred[band]
                    base_name = 'peaks_pos_%d' % band
                else:
                    band_flux = -pred[band]
                    base_name = 'peaks_neg_%d' % band
                peaks, properties = find_peaks(
                    band_flux,
                    height=np.max(np.abs(band_flux) / 5.)
                )
                num_peaks = len(peaks)

                features['%s_count' % base_name] = num_peaks

                sort_heights = np.sort(properties['peak_heights'])[::-1]
                # Measure the fractional height of the other peaks.
                for i in range(1, 3):
                    if num_peaks > i:
                        rel_height = sort_heights[i] / sort_heights[0]
                    else:
                        rel_height = np.nan
                    features['%s_frac_%d' % (base_name, (i+1))] = rel_height

        return list(features.keys()), np.array(list(features.values()))

    def extract_all_features(self):
        """Extract all features and save them to an HDF file.
        """
        all_features = []
        for i in tqdm.tqdm(range(len(self.meta_data))):
            feature_labels, features = self.extract_features(i)
            all_features.append(features)

        feature_table = pd.DataFrame(all_features, columns=feature_labels)
        feature_table.to_hdf(self.features_path, 'features', mode='w')

        # load_features does some postprocessing. Return the output of that
        # rather than the table that we saved directly.
        return self.load_features()

    def augment_object(self, idx=None, object_meta=None, object_data=None,
                       gp=None, new_ddf=None, new_redshift=None,
                       num_observations=None, full_return=False):
        """Generate an augmented version of an object for training

        The following forms of data augmentation are applied:
        - drop random observations.
        - drop large blocks of observations.
        - For galactic observations, adjust brightness (=distance).
        - For extragalactic observations, adjust redshift.
        - add noise
        """
        if idx is not None:
            orig_object_meta = self.meta_data.iloc[idx]
            orig_object_data = self.flux_data[self.flux_data['object_id'] ==
                                              orig_object_meta['object_id']]
            gp, gp_data = self.fit_gp(idx)
        else:
            orig_object_meta = object_meta
            orig_object_data = object_data

        if new_ddf is None:
            # Choose whether or not to simulate the object with DDF or WFD-like
            # conditions.
            if len(object_data) > 200:
                # Was originally in DDF, so can simulate with DDF-like
                # conditions. Note that the ratio of DDFs to WFDs here won't
                # lead to a proper DDF vs WFD distribution due to selection
                # efficiencies.
                new_ddf = np.random.binomial(1, 0.1)
            else:
                # Wasn't originally in the DDF, so has to stay out of the DDF.
                new_ddf = 0

        # Determine how many observations we would like to have.
        if num_observations is None:
            if new_ddf:
                num_observations = int(np.random.normal(330, 30))
            else:
                # I estimate the distribution of number of observations in the
                # non DDF regions with a mixture of 3 gaussian distributions.
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
                num_observations = int(np.clip(np.random.normal(mu, sigma), 50,
                                               None))

        original_redshift = orig_object_meta['hostgal_specz']
        if original_redshift == 0:
            # Galactic object. Redshift needs to stay 0.
            if new_redshift is not None and new_redshift > 0:
                raise PlasticcException("Galactic object can't be augmented to"
                                        " redshift > 0! (attempted %.3f)" %
                                        new_redshift)
            new_redshift = 0
        else:
            # Extragalactic object
            if new_redshift is None:
                # The redshift distribution is ~log-normal with a standard
                # deviation of ~0.2 log10(z). There is a systematic offset of
                # 0.2 log10(z) between the objects that have a spec-z and the
                # ones that don't in the WFD, and a systematic offset of 0.4
                # log10(z) in the DDF. Add in this offset to the test objects
                # along with some additional dispersion.
                if new_ddf:
                    adjust_logz = np.random.normal(0.4, 0.1)
                else:
                    adjust_logz = np.random.normal(0.2, 0.1)
                new_redshift = original_redshift * 10**adjust_logz
            elif new_redshift == 0:
                raise PlasticcException("Extragalactic object can't be "
                                        "augmented to redshift 0!")

        object_meta = orig_object_meta.copy()
        object_meta['hostgal_specz'] = new_redshift

        # Resample the lightcurve
        object_model = _resample_lightcurve(orig_object_data, gp, new_ddf,
                                            original_redshift, new_redshift,
                                            num_observations)

        # Add noise to the lightcurve
        object_data = _simulate_lightcurve_noise(object_model, new_ddf)

        # Model the photoz
        object_meta = _simulate_photoz(object_meta)

        # Smear the mwebv value a bit so that it doesn't uniquely identify
        # points. I leave the position on the sky unchanged (ra, dec, etc.).
        # Don't put any of those variables directly into the classifier!
        object_meta['mwebv'] *= np.random.normal(1, 0.1)

        # Update the object id by adding a random fractional offset to the id.
        # This lets us match it to the original but uniquely identify it.
        new_object_id = object_meta['object_id'] + np.random.uniform(0, 1)
        object_data['object_id'] = new_object_id
        object_meta['object_id'] = new_object_id

        object_meta['ddf'] = new_ddf

        if full_return:
            return object_meta, object_data, object_model
        else:
            return object_meta, object_data

    def augment_dataset(self, num_augments=100000, chunk=None):
        """Augment the dataset for training.

        This augmenting method attempts to produce an even distribution of each
        class at every redshift. This is accomplished by picking a random
        redshift, and then simulating one object of each target at that
        redshift. After simulating objects using this method, you shouldn't
        divide by the class counts because they will already be baked in.

        We assign the k-folding labels before augmenting to avoid biasing the
        cross-validation.
        """
        all_aug_meta = []
        all_aug_data = []

        template_metas = []
        template_datas = []
        template_gps = []
        template_redshifts = []
        template_ddfs = []

        # Load the GP fits, or create them if they aren't available.
        print("Loading the GP fits...")
        try:
            self.load_gp_fits()
        except FileNotFoundError:
            raise PlasticcException("Previous GP fits not found! Create them "
                                    "by running Dataset.fit_all_gps before "
                                    "augmenting.")

        print("Preprocessing the original dataset...")
        for target in classes[:-1]:
            all_target_metas = self.meta_data[self.meta_data['target'] ==
                                              target]
            target_metas = []
            target_datas = []
            target_gps = []
            target_redshifts = []
            target_ddfs = []

            for idx in range(len(all_target_metas)):
                object_meta = all_target_metas.iloc[idx]
                object_data = self.flux_data[self.flux_data['object_id'] ==
                                             object_meta['object_id']]
                gp = self.gp_fits[idx]

                target_metas.append(object_meta)
                target_datas.append(object_data)
                target_gps.append(gp)
                target_redshifts.append(object_meta['hostgal_specz'])
                target_ddfs.append(object_meta['ddf'] > 0)

            template_metas.append(target_metas)
            template_datas.append(target_datas)
            template_gps.append(target_gps)
            template_redshifts.append(np.array(target_redshifts))
            template_ddfs.append(np.array(target_ddfs))

        print("Augmenting the dataset...")
        sys.stdout.flush()
        for idx in tqdm.tqdm(range(num_augments)):
            # Choose whether to simulate a galactic object or not. I simulate a
            # higher fraction of galactic objects than in the test set (12%
            # there), but note that the galactic and extragalactic
            # classifications happen completely independently so this number
            # won't affect the classification in any meaningful way.
            galactic = np.random.rand() < 0.2

            # Choose whether to simulate a DDF object or not. These DDF
            # fractions are larger than in the test set (0.06% and 1%), but we
            # want to have a sizeable sample as a reference.
            if galactic:
                ddf_frac = 0.01
            else:
                ddf_frac = 0.1
            ddf = np.random.rand() < ddf_frac

            # Choose the redshift to be at. Getting the distribution here right
            # isn't that important because each redshift is effectively fit
            # independently by the classifier since we have photo-zs. I choose
            # to sample from exponential distributions whose means match the
            # mean redshifts of the WFD and DDF samples.
            if galactic:
                redshift = 0
            elif ddf:
                redshift = np.random.exponential(0.8)
            else:
                redshift = np.random.exponential(0.5)

            # For each target type, simulate an object at this redshift.
            for i in range(len(template_metas)):
                target = classes[i]

                if galactic != class_galactic[target]:
                    # Can't cross galactic/extragalactic boundaries.
                    continue

                target_metas = template_metas[i]
                target_datas = template_datas[i]
                target_gps = template_gps[i]
                target_redshifts = template_redshifts[i]
                target_ddfs = template_ddfs[i]

                # Build a mask of valid templates to use as a reference for
                # this target.

                # Can't interpolate too far in wavelength or things break. We
                # can be pickier on the minimum end because the training set is
                # a lot bluer than the test set. Filters also shift out faster
                # when you make a spectrum bluer compared to making it redder.
                # We avoid making things brighter wherever possible because
                # then the models can go crazy.
                redshift_ratio = ((1 + redshift) / (1 + target_redshifts))
                max_redshift_ratio = 1.5
                min_redshift_ratio = 0.8
                mask = ((redshift_ratio > min_redshift_ratio)
                        & (redshift_ratio < max_redshift_ratio))

                # Require a DDF template for DDF targets
                if ddf:
                    mask &= target_ddfs

                # Make sure that there are some valid templates
                if np.sum(mask) == 0:
                    continue

                use_templates = np.where(mask)[0]

                # Generate variants on the training data that are more
                # representative of the test data
                max_attempts = 50
                attempts = 0
                while True:
                    attempts += 1
                    if attempts > max_attempts:
                        # Failed to generate a lightcurve at this redshift.
                        # This is fine because selection efficiency means
                        # that some objects will never be seen at high
                        # redshifts.
                        break

                    # Choose a random element of the training set to use as the
                    # template.
                    template_id = np.random.choice(use_templates)

                    aug_meta, aug_data = self.augment_object(
                        object_meta=target_metas[template_id],
                        object_data=target_datas[template_id],
                        gp=target_gps[template_id],
                        new_ddf=ddf,
                        new_redshift=redshift
                    )

                    # Ensure that the generated object would be detected.
                    if np.sum(aug_data['detected']) < 2:
                        continue

                    # Save it and move on to the next one.
                    all_aug_meta.append(aug_meta)
                    all_aug_data.append(aug_data)
                    break

        all_aug_meta = pd.DataFrame(all_aug_meta)
        all_aug_data = pd.concat(all_aug_data, sort=False)

        new_dataset = Dataset()

        new_dataset.flux_data = all_aug_data
        new_dataset.meta_data = all_aug_meta
        if chunk is not None:
            new_dataset.dataset_name = '%s_augment_v%d_%d_%03d' % (
                self.dataset_name, self._augment_version, num_augments, chunk)
        else:
            new_dataset.dataset_name = '%s_augment_v%d_%d' % (
                self.dataset_name, self._augment_version, num_augments)

        # Save the results of the augmentation
        output_path = '%s/%s.h5' % (settings['AUGMENT_DIR'],
                                    new_dataset.dataset_name)

        new_dataset.meta_data.to_hdf(output_path, key='meta', mode='w')
        new_dataset.flux_data.to_hdf(output_path, key='df')

        return new_dataset

    def augment_dataset_old2(self, test_dataset, num_augments=100000):
        """Augment the dataset for training.

        This augmenting method uses the test dataset as a reference, and
        attempts to match the training dataset to the test dataset.

        We assign the k-folding labels before augmenting to avoid biasing the
        cross-validation.
        """
        all_aug_meta = []
        all_aug_data = []

        # Fit GPs to all data in the training set.
        template_metas = []
        template_datas = []
        template_gps = []
        print("Fitting GPs")
        for idx in tqdm.tqdm(range(len(self.meta_data))):
        # print("HACK: ONLY USING 500 OBJECTS FOR TESTING PURPOSES")
        # for idx in tqdm.tqdm(range(500)):
            object_meta = self.meta_data.iloc[idx]
            object_data = self.flux_data[self.flux_data['object_id'] ==
                                         object_meta['object_id']]
            gp, gp_data = self.fit_gp(idx)

            template_metas.append(object_meta)
            template_datas.append(object_data)
            template_gps.append(gp)

            # Keep the original training data
            all_aug_meta.append(object_meta)
            all_aug_data.append(object_data)

        # Use weights when selecting from the training set to build a less
        # unbalanced test set.
        counts = defaultdict(int)
        for meta in template_metas:
            counts[meta['target']] += 1
        ratios = {i: len(template_metas) / j for i, j in counts.items()}
        template_weights = np.array([ratios[i['target']] for i in
                                     template_metas])
        template_weights /= np.sum(template_weights)

        template_redshifts = np.array([i['hostgal_specz'] for i in
                                       template_metas])
        template_galactic = template_redshifts == 0
        template_ddf = np.array([i['ddf'] > 0 for i in template_metas])

        # Generate an augmented training set that is representative of the test
        # set
        print("Generating augmented training set")
        for idx in tqdm.tqdm(range(num_augments)):
            # Choose random element of the test set to use as a reference. We
            # scale up the number of DDF objects in the augmented set since
            # they are very rare in the test set.
            while True:
                target_id = np.random.randint(0, len(test_dataset.meta_data))
                target_meta = test_dataset.meta_data.iloc[target_id]

                if target_meta['ddf']:
                    # Keep all DDF
                    break
                else:
                    # Scale down the non-DDF so that more DDF objects end up in
                    # the final augmented training set.
                    if np.random.rand() < 0.1:
                        break

            # Build a mask of valid templates to use as a reference for this
            # target.

            # Can't interpolate too far in wavelength or things break.
            # We can be pickier on the minimum end because the training set
            # is a lot bluer than the test set. Filters also shift out
            # faster when you make a spectrum bluer compared to making it
            # redder. We avoid making things brighter wherever possible
            # because then the models can go crazy.
            redshift_ratio = ((1 + target_meta['hostgal_photoz']) /
                              (1 + template_redshifts))
            max_redshift_ratio = 1.5
            min_redshift_ratio = 0.8
            mask = ((redshift_ratio > min_redshift_ratio)
                    & (redshift_ratio < max_redshift_ratio))

            # Require galactic/extragalactic match.
            if target_meta['hostgal_photoz'] == 0:
                mask &= template_galactic
            else:
                mask &= ~template_galactic

            # Require a DDF template for DDF targets
            if target_meta['ddf']:
                mask &= template_ddf

            use_templates = np.where(mask)[0]
            use_weights = template_weights[mask]
            use_weights = use_weights / np.sum(use_weights)

            # Generate variants on the training data that are more
            # representative of the test data
            attempts = 0
            while True:
                attempts += 1
                if attempts % 1000 == 0:
                    print("WARNING: %d attempts for z=%.3f, ddf=%s" %
                          (attempts, target_meta['hostgal_photoz'],
                           target_meta['ddf']))
                # Choose a random element of the training set to use as the
                # template.
                template_id = np.random.choice(use_templates, p=use_weights)

                aug_meta, aug_data = self.augment_object(
                    object_meta=template_metas[template_id],
                    object_data=template_datas[template_id],
                    gp=template_gps[template_id],
                    new_ddf=target_meta['ddf'],
                    new_redshift=target_meta['hostgal_photoz'])

                # Ensure that the generated object would be detected.
                if np.sum(aug_data['detected']) < 2:
                    continue

                # Save it and move on to the next one.
                all_aug_meta.append(aug_meta)
                all_aug_data.append(aug_data)
                break

        all_aug_meta = pd.DataFrame(all_aug_meta)
        all_aug_data = pd.concat(all_aug_data, sort=False)

        new_dataset = Dataset()

        new_dataset.flux_data = all_aug_data
        new_dataset.meta_data = all_aug_meta
        new_dataset.dataset_name = '%s_augment_v%d_%d' % (
            self.dataset_name, self._augment_version, num_augments)

        # Save the results of the augmentation
        output_path = '%s/%s.h5' % (settings['AUGMENT_DIR'],
                                    new_dataset.dataset_name)

        new_dataset.meta_data.to_hdf(output_path, key='meta', mode='w')
        new_dataset.flux_data.to_hdf(output_path, key='df')

        return new_dataset

    def augment_dataset_old(self, num_augments=1):
        """Augment the dataset for training.

        OLD CODE: use augment_dataset which tries to properly match the
        redshift distribution of the test set.

        We need to do k-folding before augmenting or we'll bias the
        cross-validation.
        """
        all_aug_meta = []
        all_aug_data = []

        for idx in tqdm.tqdm(range(len(self.meta_data))):
            object_meta = self.meta_data.iloc[idx]
            object_data = self.flux_data[self.flux_data['object_id'] ==
                                         object_meta['object_id']]
            # Keep the original training data
            all_aug_meta.append(object_meta)
            all_aug_data.append(object_data)

            # Generate variants on the training data that are more
            # representative of the test data
            gp, gp_data = self.fit_gp(idx)
            for aug_idx in range(num_augments - 1):
                for attempt in range(5):
                    aug_meta, aug_data = self.augment_object(
                        object_meta=object_meta, object_data=object_data, gp=gp
                    )
                    if np.sum(aug_data['detected']) < 2:
                        # Not detected
                        continue

                    # Detected. Save it and move on to the next one.
                    all_aug_meta.append(aug_meta)
                    all_aug_data.append(aug_data)
                    break

        all_aug_meta = pd.DataFrame(all_aug_meta)
        all_aug_data = pd.concat(all_aug_data)

        new_dataset = Dataset()

        new_dataset.flux_data = all_aug_data
        new_dataset.meta_data = all_aug_meta
        new_dataset.dataset_name = '%s_augment_v%d_%d' % (
            self.dataset_name, self._augment_version, num_augments)

        # Save the results of the augmentation
        output_path = '%s/%s.h5' % (settings['AUGMENT_DIR'],
                                    new_dataset.dataset_name)

        new_dataset.meta_data.to_hdf(output_path, key='meta', mode='w')
        new_dataset.flux_data.to_hdf(output_path, key='df')

        return new_dataset

    def train_classifiers(self, do_fold=True):
        """Train classifiers using CV"""
        features = self.features
        y = self.meta_data['target']

        classifiers = []

        # Compute training weights.
        w = y.value_counts()
        if settings['FLAT_WEIGHT']:
            norm_class_weights = {i: 16 for i in w.index}
        elif settings['REDSHIFT_WEIGHT']:
            # Reweight in each redshift bin along with in each class bin so
            # that the classifier can't just use the redshift distribution of
            # the different classes to classify things.
            redshift_bins = np.hstack([-1, 0, np.logspace(-1, np.log10(3), 10),
                                       100])
            # redshift_bins = np.hstack([-50, np.linspace(-8, 0, 10), 5, 50])
            target_map = {j: i for i, j in enumerate(classes)}

            counts = np.zeros((len(redshift_bins), len(target_map)))

            redshift_indices = np.searchsorted(
                redshift_bins, self.meta_data['hostgal_specz']) - 1
            # redshift_indices = np.searchsorted(
                # redshift_bins, self.features['max_mag']) - 1
            class_indices = [target_map[i] for i in self.meta_data['target']]

            for redshift_index, class_index in zip(redshift_indices,
                                                   class_indices):
                counts[redshift_index, class_index] += 1

            weights = np.zeros(counts.shape)

            for redshift_index in range(len(redshift_bins)):
                # Calculate the weights for each redshift bin separately.
                bin_counts = counts[redshift_index]
                total_bin_counts = np.sum(bin_counts)

                # Add a floor to the counts to prevent absurdly high weights
                min_counts = 0.01 * total_bin_counts

                bin_counts[bin_counts < min_counts] = min_counts

                weights[redshift_index] = total_bin_counts / bin_counts

            # print(weights)

            def calc_weights(mask):
                meta_data = self.meta_data.loc[mask]
                features = self.features.loc[mask]

                redshift_indices = np.searchsorted(
                    redshift_bins, meta_data['hostgal_specz']) - 1
                # redshift_indices = np.searchsorted(
                    # redshift_bins, features['max_mag']) - 1
                class_indices = [target_map[i] for i in meta_data['target']]

                return weights[redshift_indices, class_indices]
        else:
            # Default weights
            norm_class_weights = {i: class_weights[i] * np.sum(w) / w[i] for i
                                  in w.index}

        if do_fold:
            # Do CV on folds.
            importances = pd.DataFrame()
            oof_preds = np.zeros((len(features), np.unique(y).shape[0]))
            for fold in range(5):
                print("Training fold %d." % fold)
                train_mask = self.meta_data['fold'].values != fold
                eval_mask = self.meta_data['fold'].values == fold
                train_x, train_y = features.loc[train_mask], y.loc[train_mask]
                eval_x, eval_y = features.loc[eval_mask], y.loc[eval_mask]

                if settings['REDSHIFT_WEIGHT']:
                    train_weights = calc_weights(train_mask)
                    eval_weights = calc_weights(eval_mask)
                else:
                    train_weights = train_y.map(norm_class_weights)
                    eval_weights = eval_y.map(norm_class_weights)

                classifier = fit_classifier(train_x, train_y, train_weights,
                                            eval_x, eval_y, eval_weights)

                eval_preds = classifier.predict_proba(
                    eval_x, num_iteration=classifier.best_iteration_
                )

                oof_preds[eval_mask, :] = eval_preds

                imp_df = pd.DataFrame()
                imp_df['feature'] = features.columns
                imp_df['gain'] = classifier.feature_importances_
                imp_df['fold'] = fold + 1
                importances = pd.concat([importances, imp_df], axis=0,
                                        sort=False)

                classifiers.append(classifier)

            # Full sample
            print('MULTI WEIGHTED LOG LOSS : %.5f ' %
                  multi_weighted_logloss(y_true=y, y_preds=oof_preds))

            # Original sample only (no augments)
            original_mask = (self.meta_data['object_id'] % 1) == 0
            if np.sum(original_mask) > 0:
                # Can only do this for augments that include the original
                # training sample...
                print('BASE WEIGHTED LOG LOSS: %.5f' %
                      multi_weighted_logloss(y.values[original_mask],
                                             oof_preds[original_mask]))

            # Build a pandas dataframe with the out of fold predictions
            oof_preds_df = pd.DataFrame(data=oof_preds,
                                        columns=['class_%d' % i for i in
                                                 classes[:-1]])
            self.fit_preds = oof_preds_df
        else:
            # Train a single classifier.
            train_weights = y.map(norm_class_weights)
            classifier = fit_classifier(features, y, train_weights,
                                        n_estimators=700)

            importances = pd.DataFrame()
            importances['feature'] = features.columns
            importances['gain'] = classifier.feature_importances_
            importances['fold'] = 1
            classifiers.append(classifier)

        # Save results of the fit internally, and return the fitted
        # classifiers.
        self.importances = importances

        return classifiers

    def fit_salt(self, *args, full_return=False, verbose=False, **kwargs):
        """Perform a SALT2 fit on an object using sncosmo"""
        import sncosmo
        gp_data = self.predict_gp(*args, **kwargs)
        fit_data = Table({
            'time': gp_data['times'],
            'band': ['lsst%s' % ['u', 'g', 'r', 'i', 'z', 'y'][i] for i in
                     gp_data['bands']],
            'flux': gp_data['fluxes'],
            'fluxerr': gp_data['flux_errs'],
            'zp': [20. for i in range(len(gp_data['bands']))],
            'zpsys': ['ab' for i in range(len(gp_data['bands']))],
        })

        model = sncosmo.Model(source='salt2-extended')
        guess_z = gp_data['meta']['hostgal_photoz']
        guess_z_err = gp_data['meta']['hostgal_photoz_err']

        weights = []
        guess_maxes = []
        for band in range(6):
            mask = gp_data['bands'] == band
            band_guess_max = gp_data['times'][mask][
                np.argmax(gp_data['fluxes'][mask])]

            guess_maxes.append(band_guess_max)

            s2ns = gp_data['fluxes'][mask] / gp_data['flux_errs'][mask]
            # Weight is total s/n in each band
            weights.append(np.sqrt(np.sum(s2ns**2)))
        weights = np.array(weights)
        guess_maxes = np.array(guess_maxes)

        guess_max = weighted_median(guess_maxes, weights)

        bounds = {
            'x1': (-5, 5),
            'c': (-1, 3),
        }

        fit_params = ['t0', 'x0', 'x1', 'c']

        if guess_z > 1e-5:
            model['z'] = guess_z
            bounds['z'] = (max(1e-5, guess_z-3*guess_z_err),
                           max(1e-5, guess_z+3*guess_z_err))
            fit_params.append('z')
        else:
            # Fix z for galactic objects
            model['z'] = 1e-5

        model['t0'] = guess_max
        model['x0'] = np.max(gp_data['fluxes']) * 1e-4

        if verbose:
            print("Guess z:    ", guess_z)
            print("Guess z err:", guess_z_err)
            try:
                print("True z:     ", gp_data['meta']['hostgal_specz'])
            except KeyError:
                pass
            print("Bounds:     ", bounds)

        result, fitted_model = sncosmo.fit_lc(
            fit_data,
            model,
            fit_params,
            bounds=bounds,
            guess_t0=False,
            guess_z=False,
            guess_amplitude=False
        )

        if verbose:
            print("---")
            print("    guess    fit")
            if guess_z > 1e-5:
                print("z:  %8.3f %8.3f" % (model['z'], fitted_model['z']))
            print("t0: %8.3f %8.3f" % (model['t0'], fitted_model['t0']))
            print("x0: %8.3f %8.3f" % (model['x0'], fitted_model['x0']))
            print("x1: %8.3f %8.3f" % (model['x1'], fitted_model['x1']))
            print("c:  %8.3f %8.3f" % (model['c'], fitted_model['c']))

        if full_return:
            return fit_data, result, fitted_model
        else:
            return dict(zip(fitted_model.param_names, fitted_model.parameters))

    def fit_all_salt(self):
        """Fit all lightcurves with SALT2 and save the results to an HDF file.
        """
        all_salt_fits = []
        for i in tqdm.tqdm(range(len(self.meta_data))):
            salt_fit = self.fit_salt(i)
            all_salt_fits.append(salt_fit)

        salt_fits_table = pd.DataFrame(all_salt_fits)
        salt_fits_table.to_hdf(self.salt_fits_path, 'salt_fits', mode='w')

        return salt_fits_table

    def plot_salt(self, *args, **kwargs):
        import sncosmo
        fit_data, result, fitted_model = self.fit_salt(*args, full_return=True,
                                                       **kwargs)
        sncosmo.plot_lc(fit_data, model=fitted_model, errors=result.errors)


def weighted_median(x, weights):
    order = np.argsort(x)
    order_x = x[order]
    order_weights = weights[order]

    total_weight = np.sum(order_weights)
    med_loc = np.argwhere(np.cumsum(order_weights) > total_weight / 2)[0][0]
    return order_x[med_loc]


def save_classifiers(classifiers, path):
    """Save a list of classifiers to a path as a pickle file"""
    with open(path, 'wb') as fout:
        pickle.dump(classifiers, fout)


def load_classifiers(path):
    """Load a list of classifiers from a path"""
    with open(path, 'rb') as fin:
        classifiers = pickle.load(fin)

    return classifiers


def _resample_lightcurve(object_data, gp, new_ddf, original_redshift,
                         new_redshift, num_observations):
    """Resample a lightcurve as part of the augmenting procedure

    This step applies several transformations to the lightcurve and drops
    observations where appropriate.

    Parameters
    ----------
    object_data : pandas DataFrame
        The flux data for the object that will be augmented.
    gp : wrapped george gaussian process.
        A george Gaussian process object fit to the original object flux,
        as returned from fit_gp.
    new_ddf : bool
        If True, the lightcurve is sampled as if it were in the DDF sample. If
        False, the lightcurve is sampled as if it were in the WFD sample.
    original_redshift : float
        The original redshift of the lightcurve.
    new_redshift : float
        The new redshift value to shift the lightcurve to.
    num_observations : int
        How many observations to have for the new lightcurve.

    Returns
    -------
    pandas DataFrame
        The new flux data for the resampled lightcurve.
    """
    # Figure out where to sample the new lightcurve at. Start with a copy
    # of the original flux data and the reference MJDs.
    ref_object_data = object_data
    object_data = ref_object_data.copy()
    object_data['ref_mjd'] = object_data['mjd'].copy()

    if (original_redshift == 0) != (new_redshift == 0):
        # Can't make extragalactic objects galactic and vice-versa
        raise PlasticcException("Can't flip extragalactic/galactic in "
                                "augmentation! z=%.3f->z=%.3f" %
                                (original_redshift, new_redshift))

    # Shift and scale the times for the new observations.
    if original_redshift != 0:
        # For extragalactic objects, redshift the observing times.
        # Figure out a reference point to scale around so that we don't scale
        # the template out of the observing window too often.
        orig_ref_mjd = (object_data['mjd'].iloc[
            np.argmax(object_data['flux'].values)])

        # Shift the new reference point slightly.
        new_ref_mjd = orig_ref_mjd + np.random.uniform(-50, 50)

        new_mjd = (
            new_ref_mjd +
            (object_data['mjd'] - orig_ref_mjd) * (1 + new_redshift) /
            (1 + original_redshift)
        )
        object_data['mjd'] = new_mjd
    else:
        # For galactic objects, add a small shift to the observation times.
        object_data['mjd'] += np.random.uniform(-50, 50)

    # Drop a block of observations corresponding to the typical width of a
    # season to create lightcurves with large missing chunks.
    block_width = 250
    block_start = np.random.uniform(start_mjd-block_width, end_mjd)
    block_end = block_start + block_width
    block_mask = ((object_data['mjd'] < block_start) |
                  (object_data['mjd'] > block_end))
    object_data = object_data[block_mask].copy()

    # Drop observations that are outside of the observing window after all of
    # these procedures. We leave a bit of a buffer to get better baselines for
    # background estimation.
    object_data = object_data[
        (object_data['mjd'] > start_mjd - pad).values &
        (object_data['mjd'] < end_mjd + pad).values
    ].copy()

    # At high redshifts, we need to fill in the lightcurve to account for the
    # fact that there is a lower observation density compared to lower
    # redshifts.
    num_fill = int(num_observations * (
        (1 + new_redshift) / (1 + original_redshift) - 1))
    if num_fill > 0:
        new_indices = np.random.choice(object_data.index, num_fill,
                                       replace=True)
        new_rows = object_data.loc[new_indices]

        # Tweak the MJDs of the new rows slightly.
        # UPDATE: Don't do this. Tweaking the MJDs just gives the GP the
        # potential to do bad things, especially for short timescale targets
        # like kilonovae. All that we really care about is getting the correct
        # signal-to-noise for each bin.
        # tweak_scale = 2
        # mjd_tweaks = np.random.uniform(-tweak_scale, tweak_scale, num_fill)
        # new_rows['mjd'] += mjd_tweaks BUG: ADD IN REDSHIFT HERE!!!
        # new_rows['ref_mjd'] += mjd_tweaks

        # Choose bands randomly. We only move by a max of one band to avoid
        # breaking things.
        band_offsets = np.random.randint(-1, 2, num_fill)
        new_rows['passband'] = np.clip(new_rows['passband'] + band_offsets, 0,
                                       num_passbands-1)

        object_data = pd.concat([object_data, new_rows])

    # Drop back down to the target number of observations. Having too few
    # observations is fine, but having too many is not. We always drop at least
    # 10% of observations to get some shakeup of the lightcurve.
    num_drop = int(max(len(object_data) - num_observations,
                       0.1*num_observations))
    drop_indices = np.random.choice(
        object_data.index, num_drop, replace=False
    )
    object_data = object_data.drop(drop_indices).copy()

    # Compute the fluxes from the GP at the new observation points.
    new_wavelengths = np.array([band_wavelengths[int(i)] for i in
                                object_data['passband']])
    eval_wavelengths = (new_wavelengths * (1 + original_redshift) /
                        (1 + new_redshift))
    pred_x_data = np.vstack([object_data['ref_mjd'] - start_mjd,
                             eval_wavelengths]).T
    new_fluxes, new_fluxvars = gp(pred_x_data, return_var=True)

    # Set a floor on the error at the signal-to-noise of the reference
    # observations. This helps avoid creating significant negative fluxes for
    # brighter observations due to modeling errors which deteriorate the
    # classifier performance.
    # UPDATE: Shouldn't be necessary with the GP error. I wasn't propagating
    # that properly before, but it should be fixed now.
    # ref_s2n = object_data['flux'] / object_data['flux_err']
    # ref_err = new_fluxes / ref_s2n

    object_data['flux'] = new_fluxes
    # object_data['flux_err'] = np.min([ref_err, np.sqrt(new_fluxvars)], axis=0)
    object_data['flux_err'] = np.sqrt(new_fluxvars)

    # Update the brightness of the new observations.
    if original_redshift == 0:
        # UPDATE: I changed the error calculation to set a floor on the noise
        # at the level of the input. A normal distribution should be fine now
        # since everything should still be fine for brighter objects.
        # Adjust brightness for galactic objects. We only want to make things
        # fainter to avoid issues when the GP is untrustworthy and the errors
        # blow up when things get brighter.
        adjust_mag = np.random.normal(0, 0.5)
        # adjust_mag = np.random.lognormal(-1, 0.5)
        adjust_scale = 10**(-0.4*adjust_mag)
    else:
        # Adjust brightness for extragalactic objects. We simply follow the
        # Hubble diagram.
        delta_distmod = (cosmo.distmod(original_redshift) -
                         cosmo.distmod(new_redshift)).value
        adjust_scale = 10**(0.4*delta_distmod)

    object_data['flux'] *= adjust_scale

    # We have the resampled models! Note that there is no error added in yet,
    # so we set the detected flags to default values and clean up.
    object_data['detected'] = 1
    object_data.reset_index(inplace=True, drop=True)

    return object_data


def _simulate_lightcurve_noise(object_data, ddf):
    """Simulate noise for a lightcurve.

    Add noise to match what we see in the real data. Each band has a different
    noise level. I fit a log-normal distribution to the uncertainties in the
    WFD and DDF subsets separately. Note that I don't handle Poisson error or
    template subtraction error in these estimates, so be careful with features
    directly related to the noise or you could end up overfitting bright
    training data. There are also plenty of weird outlier data points in the
    PLAsTiCC data, probably due to things like cosmic rays. Those don't matter
    for my algorithm because they always have large uncertainties so the
    Gaussian process simply ignores them.

    Parameters
    ----------
    object_data : pandas DataFrame
        Noise-free observations for an object that that we will be adding noise
        to.
    ddf : bool
        If True, the lightcurve is treated as if it were in the DDF sample. If
        False, the lightcurve is treated as if it were in the WFD sample.

    Returns
    -------
    pandas DataFrame
        The new flux data for the noised-up lightcurve.
    """
    # Make a copy so that we don't modify the original array.
    object_data = object_data.copy()

    if len(object_data) == 0:
        # No data, skip
        return object_data

    if ddf:
        band_noises = {
            0: (0.68, 0.26),
            1: (0.25, 0.50),
            2: (0.16, 0.36),
            3: (0.53, 0.27),
            4: (0.88, 0.22),
            5: (1.76, 0.23),
        }
    else:
        band_noises = {
            0: (2.34, 0.43),
            1: (0.94, 0.41),
            2: (1.30, 0.41),
            3: (1.82, 0.42),
            4: (2.56, 0.36),
            5: (3.33, 0.37),
        }

    # Calculate the new noise levels using a lognormal distribution for
    # each band.
    lognormal_parameters = np.array([band_noises[i] for i in
                                     object_data['passband']])
    add_stds = np.random.lognormal(lognormal_parameters[:, 0],
                                   lognormal_parameters[:, 1])

    noise_add = np.random.normal(loc=0.0, scale=add_stds)
    object_data['flux'] += noise_add
    object_data['flux_err'] = np.sqrt(object_data['flux_err']**2 + add_stds**2)

    # The 'detected' threshold doesn't seem to be a simple function of
    # signal-to-noise. For now, I model it with an error function on
    # the signal-to-noise, but there is probably a proper way to
    # determine if there was a dectection or not...
    s2n = np.abs(object_data['flux']) / object_data['flux_err']
    prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.
    object_data['detected'] = np.random.rand(len(s2n)) < prob_detected

    return object_data


def _simulate_photoz(object_meta):
    """Simulate the photoz determination for a lightcurve.

    I estimate the photoz with a Gaussian mixture model that was approximated
    from the real data. There is a narrow core with broader wings along with a
    weird group that clumps at z=2.5 regardless of the true redshift.

    Parameters
    ----------
    object_meta : pandas Series
        Original object meta data

    Returns
    -------
    pandas Series
        The object meta data with an updated photoz estimate.
    """
    # Make a copy of the object metadata so that we don't affect the input.
    object_meta = object_meta.copy()

    if object_meta['hostgal_specz'] == 0:
        # Galactic object. For the PLAsTiCC challenge, the reshift is always 0.
        # Leave things as is.
        return object_meta

    new_photoz = -1
    while new_photoz < 0:
        gauss_choice = np.random.choice(4, p=[0.73, 0.14, 0.1, 0.03])
        if gauss_choice == 0:
            # Good photoz
            photoz_std = (np.random.normal(0.009, 0.001) +
                          np.random.gamma(1.2, 0.01))
            photoz_err = np.random.normal(0, photoz_std)
        elif gauss_choice == 1:
            # photoz is okish, but error is garbage
            photoz_std = np.random.gamma(2., 0.2)
            photoz_err = np.random.normal(0, 0.05)
        elif gauss_choice == 2:
            # photoz is offset by ~0.2.
            photoz_std = np.random.gamma(2., 0.2)
            photoz_err = 0.2 + np.random.normal(0, 0.1)
        elif gauss_choice == 3:
            # garbage photoz. For some reason this just outputs a
            # number around 2.5
            photoz_std = np.random.uniform(0.1, 1.5)
            photoz_err = np.random.normal(2.5, 0.2)

        if gauss_choice != 3:
            new_photoz = object_meta['hostgal_specz'] + photoz_err
        else:
            # Catastrophic failure, just get z=2.5 regardless of the
            # true redshift.
            new_photoz = photoz_err

    object_meta['hostgal_photoz'] = new_photoz
    object_meta['hostgal_photoz_err'] = photoz_std
    object_meta['distmod'] = cosmo.distmod(new_photoz).value

    return object_meta
