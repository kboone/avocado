# avocado script augmenting spcc - requires spcc_sn_data.txt file
import numpy as np
import pandas as pd
import astronomical_object
import augment
from scipy.special import erf
import string
import math

from .instruments import band_central_wavelengths
from .utils import settings, logger

class SPCC_SN_data:
    """Metadata for the SPCC dataset"""
    
    def __init__(self):
        
        # load spcc sn data
        spcc_sn_data = pd.read_csv('../data/spcc_sn_data.txt',delimiter=' ', names=['spcc_names', 'spcc_types', 'spcc_mags', 'spcc_photo_z', 'spcc_photo_z_err', 'spcc_spec_z'])
        self.spcc_names = spcc_sn_data['spcc_names'].values
        self.spcc_types = spcc_sn_data['spcc_types'].values
        self.spcc_mags = spcc_sn_data['spcc_mags'].values
        self.spcc_photo_z = spcc_sn_data['spcc_photo_z'].values
        self.spcc_photo_z_err = spcc_sn_data['spcc_photo_z_err'].values
        self.spcc_spec_z = spcc_sn_data['spcc_spec_z'].values
        
        self.binned_data = None
        
    def get_binned_data(self):
        """Bin SPCC r-magnitudes by redshift to later check whether augmented
        objects are similar and likely to cover the same parameter space
        
        Returns
        -------
        binned_data : 2d list
            SPCC magnitudes binned by redshift
        """
        if self.binned_data is None:
            spcc_z_bins = [[] for i in range(11)]
            
            # there's probably a better way to do this, but this works and is fast
            for i,z in enumerate(self.spcc_spec_z):
                if float(z) <= 0.1:
                    spcc_z_bins[0].append(self.spcc_mags[i])
                elif float(z) <= 0.2:
                    spcc_z_bins[1].append(self.spcc_mags[i])
                elif float(z) <= 0.3:
                    spcc_z_bins[2].append(self.spcc_mags[i])
                elif float(z) <= 0.4:
                    spcc_z_bins[3].append(self.spcc_mags[i])
                elif float(z) <= 0.5:
                    spcc_z_bins[4].append(self.spcc_mags[i])
                elif float(z) <= 0.6:
                    spcc_z_bins[5].append(self.spcc_mags[i])
                elif float(z) <= 0.7:
                    spcc_z_bins[6].append(self.spcc_mags[i])
                elif float(z) <= 0.8:
                    spcc_z_bins[7].append(self.spcc_mags[i])
                elif float(z) <= 0.9:
                    spcc_z_bins[8].append(self.spcc_mags[i])
                elif float(z) <= 1.0:
                    spcc_z_bins[9].append(self.spcc_mags[i])
                elif float(z) <= 1.1:
                    spcc_z_bins[10].append(self.spcc_mags[i])
            
            # remove outlier type ii supernova,which skews distribution of
            #augmented objects - as this is a strange light curve with only
            #a couple very faint points in r band, seen in plot of r-mag vs z
            spcc_z_bins[2].remove(27.119)
            
            self.binned_data = spcc_z_bins
            
        return self.binned_data
    
    def get_augmentation_list(self, training_list, add_faint=False, training_faint_list=''):
        """Read list of SPCC training objects to augment from a file
        If already created augmented objects from a sample (e.g. mag-limited),
        can augment additional supernovae (e.g. a fainter sample)
        
        Parameters
        ----------
        training_list : string
            Directory of file containing list of SPCC objects to augment
        add_faint : bool
            If true, need to specify training_faint_list, for augmenting just
            additional supernovae not included in the original training_list
        training_faint_list : string
            Directory of file with list of training objects including fainter sample
        
        Returns
        -------
        training_objects : numpy.array
            Array of objects for augmenting
        """
        training_objects = np.loadtxt(training_list, dtype=str)
        
        if add_faint:
            # read list which comprises training_objects and additional objects
            training_add_faint = np.loadtxt(training_faint_list, dtype=str)
            # get just the additional objects
            faint_objs = np.setdiff1d(training_add_faint, training_objects)
            # redefine training_objects array so that we only use this list
            training_objects = np.asarray(faint_objs)
        
        return training_objects
    
    def get_photoz_reference(self):
        """Get the reference for photo-z estimation from the SPCC dataset
        
        Returns
        -------
        photoz_reference : numpy ndarray
            Nx3 array with reference photo-zs for each entry with a spec-z.
            The columns are spec-z, photo-z, photo-z error
        """
        
        self.photoz_reference = np.vstack([self.spcc_spec_z, self.spcc_photo_z, self.spcc_photo_z_err]).T
        
        return self.photoz_reference
    
    def load_reference_object(self, sn):
        """Load reference object to augment from
        
        Parameters
        ----------
        sn : string
            Original SPCC supernova ID
        
        Returns
        -------
        reference_object : :class:'AstronomicalObject'
            The object to be used as a reference for augmentation
        """
        
        sn_index = list(self.spcc_names).index(sn)
        obj_mag = self.spcc_mags[sn_index]
        photo_z = self.spcc_photo_z[sn_index]
        photo_z_err = self.spcc_photo_z_err[sn_index]
        spec_z = self.spcc_spec_z[sn_index]
        sn_type = self.spcc_types[sn_index]
        
        mjd = []
        flt = []
        flux = []
        flux_err = []
        # it should be specified here where the SPCC data is stored
        #with open('../../SIMGEN_PUBLIC_DES/'+sn, 'r') as f:
        with open('/dir/to/data/SIMGEN_PUBLIC_DES/'+sn, 'r') as f:
            obj_name = str(sn)
            for line in f:
                line_list = line.split()
                if 'OBS:' in line and len(line_list)>2:
                    mjd.append(float(line_list[1]))
                    flt.append('des'+line_list[2])
                    flux.append(float(line_list[4]))
                    flux_err.append(float(line_list[5]))

        obs_dict = {'time': mjd, 'band': flt, 'flux': flux, 'flux_error': flux_err}
        observations = pd.DataFrame(obs_dict)
        metadata = {'object_id': obj_name, 'object_r_mag': obj_mag, 'host_photoz': photo_z, 'host_photoz_error': photo_z_err, 'host_specz': spec_z, 'redshift': spec_z, 'class': sn_type}
        
        reference_object = astronomical_object.AstronomicalObject(metadata, observations)
        
        return reference_object
    
    def passed_criteria(self, augmented_object):
        """Make sure augmented object passes criteria; there should be
        multiple r band observations with positive peak flux and it should
        exist within the magnitude and redshift bounds of the dataset
        
        Parameters
        ----------
        augmented_object : :class:'AstronomicalObject'
            Created augmented object
        
        Returns
        -------
        bool
            True or False for passing criteria
        """
        r_fluxes = []
        for i,flt in enumerate(augmented_object.observations['band']):
            if flt=='desr':
                r_fluxes.append(augmented_object.observations['flux'][i])
        if len(r_fluxes)==0:
            return False
        max_flux = np.amax(r_fluxes)
        if max_flux<0:
            return False
        
        r_mag = -2.5*math.log10(max_flux)+27.5 # zero point from sn data
        
        augmented_object.metadata['object_r_mag'] = r_mag
        
        augmented_z = augmented_object.metadata['host_specz']
        
        # remove augmented objects outside z-mag spaces.
        #in reality we will have mag and z info
        binned_data = self.get_binned_data()
        for i,z_bin in enumerate(binned_data):
            if augmented_z > i/10 and augmented_z <= (i/10+0.1):
                if r_mag > np.amin(z_bin) and r_mag <= np.amax(z_bin):
                    return True
                else:
                    return False
            else:
                if augmented_z > (1.1):
                    return False
      
    def save_augmented_object(self, augmented_object, augment_dir):
        """Create .DAT file of the augmented light curve that includes
        the necessary info for photometric classification, following
        the same formatting used in the original SPCC
        
        Parameters
        ----------
        augmented_object : :class:'AstronomicalObject'
            Created augmented object
        augment_dir : string
            Directory in which to save augmented object
        """
        augmented_mjd = augmented_object.observations['time']
        augmented_flux = augmented_object.observations['flux']
        augmented_flux_err = augmented_object.observations['flux_error']
        augmented_bands = augmented_object.observations['band']
        renamed_augmented_bands = []
        for filt in augmented_bands:
            renamed_augmented_bands.append(filt.split('des')[1])

        augmented_z = str(augmented_object.metadata['host_specz'])
        obj_class = augmented_object.metadata['class']
        obj_id = augmented_object.metadata['object_id']
        r_mag = str(augmented_object.metadata['object_r_mag'])
        
        aug_obj_file = open(augment_dir+'/'+obj_id, 'w')
        aug_obj_file.write('SIM_REDSHIFT: '+augmented_z+'\n')
        aug_obj_file.write('SIM_COMMENT: '+'SN Type = '+obj_class+'\n')
        aug_obj_file.write('R-MAG = '+r_mag+'\n')
        for i,val in enumerate(augmented_mjd):
            aug_obj_file.write('OBS: '+str(val)+' '+str(renamed_augmented_bands[i])+' '+'0'+' '+str(augmented_flux[i])+' '+str(augmented_flux_err[i])+'\n')
        aug_obj_file.close()
        
        
class SpccAugmentor(augment.Augmentor):
    """Subclass of the avocado augmentor for the SPCC dataset.
    Most methods implemented here are changed slightly to the original
    ones in avocado
    """
    
    def __init__(self, **cosmology_kwargs):
        super().__init__()
    
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
        # no galactic objects (z=0) in spcc
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
        aug_redshift = np.exp(np.random.uniform(np.log(min_redshift), np.log(max_redshift)))

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

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        Returns
        =======
        augmented_metadata : dict
            The augmented metadata
        """
        # no need for ddfs in spcc, or mwebv etc since we aren't using this for
        #classification, just making new objects, so just make copy to start from
        augmented_metadata = reference_object.metadata.copy() 
        self._augment_redshift(reference_object, augmented_metadata)

        return augmented_metadata
    
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
        photoz_reference = SPCC_SN_data().get_photoz_reference()

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
    
    def _choose_target_observation_count(self, augmented_metadata):
        """Choose the target number of observations for a new augmented light
        curve.

        Implemented for the SPCC dataset

        Parameters
        ==========
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        target_observation_count : int
            The target number of observations in the new light curve.
        """
        # number of light curve observations in SPCC modelled well with a two-peaked distribution
        gauss_choice = np.random.choice(2, p=[0.25,0.75])
        if gauss_choice == 0:
            mu = 51
            sigma = 15
        elif gauss_choice == 1:
            mu = 110
            sigma = 24
        target_observation_count = int(np.clip(np.random.normal(mu, sigma), 16, None)) # choose 16 as this is the minimum number of observations in a SPCC light curve
        
        return target_observation_count
    
    def _simulate_light_curve_uncertainties(self, observations, augmented_metadata):
        """Simulate the observation-related noise for a light curve.
        
        Implemented for the SPCC dataset

        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process. These observations have model flux uncertainties listed
            that should be included in the final uncertainties.
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        observations : pandas.DataFrame
            The observations with uncertainties added.
        """
        observations = observations.copy()
        
        band_noises = {'desg': (1.459, 0.889), 'desr': (0.807, 0.891), 'desi': (1.305, 0.801), 'desz': (1.285, 0.737)}

        # Calculate the new noise levels using a lognormal distribution for
        # each band.
        lognormal_parameters = np.array([band_noises[i] for i in observations['band']])

        add_stds = np.random.lognormal(lognormal_parameters[:, 0], lognormal_parameters[:, 1])

        noise_add = np.random.normal(loc=0.0, scale=add_stds)
        
        observations['flux'] += noise_add
        observations['flux_error'] = np.sqrt(observations['flux_error']**2 + add_stds**2)
        # for not including GP error:
        #observations['flux_error'] = abs(add_stds)
        
        return observations
    
    def _simulate_detection(self, observations, augmented_metadata):
        """Simulate the detection process for a light curve.

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
        s2n = np.abs(observations['flux']) / observations['flux_error']
        prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.
        observations['detected'] = np.random.rand(len(s2n)) < prob_detected

        pass_detection = np.sum(observations['detected']) >= 2

        return observations, pass_detection
    
    def _choose_sampling_times(self, reference_object, augmented_metadata, max_time_shift=50, window_padding=100, drop_fraction=0.1):
        """Choose the times at which to sample for a new augmented object.
        
        Implemented for the SPCC dataset. No need to drop large observation blocks.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.
        augmented_metadata : dict
            The augmented metadata
        max_time_shift : float (optional)
            The new sampling times will be shifted by up to this amount
            relative to the original ones.
        window_padding : float (optional)
            Observations outside of a window bounded by the first and last
            observations in the reference objects light curve with a padding
            specified by this parameter will be dropped.
        drop_fraction : float (optional)
            This fraction of observations will always be dropped when creating
            the augmented light curve.

        Returns
        =======
        sampling_times : pandas Dataframe
            A pandas Dataframe that has the following columns:

            - time : the times of the simulated observations.
            - band : the bands of the simulated observations.
            - reference_time : the times in the reference light curve that
              correspond to the times of the simulated observations.
        """
        # Figure out the target number of observations to have for the new
        # lightcurve.
        target_observation_count = self._choose_target_observation_count(augmented_metadata)

        # Start with a copy of the original times and bands.
        reference_observations = reference_object.observations
        sampling_times = reference_observations[['time', 'band']].copy()
        sampling_times['reference_time'] = sampling_times['time'].copy()

        start_time = np.min(sampling_times['time'])
        end_time = np.max(sampling_times['time'])

        # If the redshift changed, shift the time of the observations.
        augmented_redshift = augmented_metadata['redshift']
        reference_redshift = reference_object.metadata['host_specz']
        redshift_scale = (1 + augmented_redshift) / (1 + reference_redshift)

        if augmented_redshift != reference_redshift:
            # Shift relative to an approximation of the peak flux time so that
            # we generally keep the interesting part of the light curve in the
            # frame.
            ref_peak_time = reference_observations['time'].iloc[np.argmax(reference_observations['flux'].values)]
            
            sampling_times['time'] = (ref_peak_time + redshift_scale * (sampling_times['time'] - ref_peak_time))

        # Shift the observations forward or backward in time by a small
        # amount.
        sampling_times['time'] += np.random.uniform(-max_time_shift, max_time_shift)

        # Drop observations that are outside of the observing window after all
        # of these procedures. We leave a bit of a buffer to get better
        # baselines for background estimation.
        sampling_times = sampling_times[(sampling_times['time'] > start_time - window_padding).values & (sampling_times['time'] < end_time + window_padding).values].copy()

        # At high redshifts, we need to fill in the light curve to account for
        # the fact that there is a lower observation density compared to lower
        # redshifts.
        num_fill = int(target_observation_count * (redshift_scale - 1))
        
        if num_fill > 0:
            new_indices = np.random.choice(sampling_times.index, num_fill, replace=True)
            new_rows = sampling_times.loc[new_indices]
            
            # Choose new bands randomly.
            new_rows['band'] = np.random.choice(reference_object.bands, num_fill, replace=True)

            sampling_times = pd.concat([sampling_times, new_rows])
        
        # Drop back down to the target number of observations. Having too few
        # observations is fine, but having too many is not. We always drop at
        # least 10% of observations to get some shakeup of the light curve.
        num_drop = int(max(len(sampling_times) - target_observation_count, drop_fraction * target_observation_count))
    
        drop_indices = np.random.choice(sampling_times.index, num_drop, replace=False)
        sampling_times = sampling_times.drop(drop_indices).copy()

        sampling_times.reset_index(inplace=True, drop=True)

        return sampling_times
    
    def _resample_light_curve(self, reference_object, augmented_metadata):
        """Resample a light curve as part of the augmenting procedure
        This uses the Gaussian process fit to a light curve to generate new
        simulated observations of that light curve.
        In some cases, the light curve that is generated will be accidentally
        shifted out of the frame, or otherwise missed. If that is the case, the
        light curve will automatically be regenerated with the same metadata
        until it is either detected or until the number of tries has exceeded
        settings['augment_retries'].
        
        Parameters
        ----------
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.
        augmented_metadata : dict
            The augmented metadata
        Returns
        -------
        augmented_observations : pandas.DataFrame
            The simulated observations for the augmented object. If the chosen
            metadata leads to an object that is too faint or otherwise unable
            to be detected, None will be returned instead.
        """
        
        gp  = reference_object.get_default_gaussian_process()
        
        # Figure out where to sample the augmented light curve at.
        observations = self._choose_sampling_times(reference_object, augmented_metadata)
        
        # Compute the fluxes from the GP at the augmented observation
        # times.
        new_redshift = augmented_metadata['redshift']
        reference_redshift = reference_object.metadata['host_specz']
        redshift_scale = (1 + new_redshift) / (1 + reference_redshift)

        new_wavelengths = np.array([band_central_wavelengths[i] for i in observations['band']])
        eval_wavelengths = new_wavelengths / redshift_scale
        pred_x_data = np.vstack([observations['reference_time'], eval_wavelengths]).T
        new_fluxes, new_fluxvars = gp(pred_x_data, return_var=True)

        observations['flux'] = new_fluxes
        observations['flux_error'] = np.sqrt(new_fluxvars)

        # Update the brightness of the new observations. If the
        # 'augment_brightness' key is in the metadata, we add that in
        # magnitudes to the augmented object.
        augment_brightness = augmented_metadata.get('augment_brightness', 0)
        adjust_scale = 10**(-0.4*augment_brightness)

        # All objects in spcc are extragalactic - adjust brightness following
        # the Hubble diagram.
        delta_distmod = (self.cosmology.distmod(reference_redshift) - self.cosmology.distmod(new_redshift)).value
        adjust_scale *= 10**(0.4*delta_distmod)

        observations['flux'] *= adjust_scale
        observations['flux_error'] *= adjust_scale

        # Save the model flux and flux error
        observations['model_flux'] = observations['flux']
        observations['model_flux_error'] = observations['flux_error']

        # Add in light curve noise. This is survey specific and must be
        # implemented in subclasses.
        observations = self._simulate_light_curve_uncertainties(observations, augmented_metadata)

        # Simulate detection
        observations, pass_detection = self._simulate_detection(observations, augmented_metadata)
        
        # If our light curve passes detection thresholds, we're done!
        if pass_detection:
            return observations

        # Failed to generate valid observations.
        return None
    
    def augment_object(self, reference_object, force_success=True):
        """Generate an augmented version of an object.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.
        force_success : bool
            If True, then if we fail to generate an augmented light curve for a
            specific set of augmented parameters, we choose a different set of
            augmented parameters until we eventually get an augmented light
            curve. This is useful for debugging/interactive work, but when
            actually augmenting a dataset there is a massive speed up to
            ignoring bad light curves without a major change in classification
            performance.

        Returns
        =======
        augmented_object : :class:`AstronomicalObject`
            The augmented object. If force_success is False, this can be None.
        """
        # Create a new object id for the augmented object. We choose a random
        # string to add on to the end of the original object id that is very
        # unlikely to have collisions.
        ref_object_id = reference_object.metadata['object_id']
        random_str = ''.join(np.random.choice(list(string.ascii_letters), 10))
        new_object_id = '%s_aug_%s.DAT' % (ref_object_id.split('.')[0], random_str)
        
        while True:
            # Augment the metadata. The details of how this should work is
            # survey specific, so this must be implemented in subclasses.
            augmented_metadata = self._augment_metadata(reference_object)
            augmented_metadata['object_id'] = new_object_id
            augmented_metadata['reference_object_id'] = ref_object_id
            
            # Generate an augmented light curve for this augmented metadata.
            observations = self._resample_light_curve(reference_object, augmented_metadata)

            if observations is not None:
                # Successfully generated a light curve.
                augmented_object = astronomical_object.AstronomicalObject(augmented_metadata, observations)
                return augmented_object
            elif not force_success:
                # Failed to generate a light curve, and we aren't retrying
                # until we are successful.
                return None
            else:
                logger.warn("Failed to generate a light curve for redshift %.2f. Retrying." % augmented_metadata['redshift'])
