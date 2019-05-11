import numpy as np
import string

class Augmentor():
    """Class used to augment a dataset.

    This class takes :class:`AstronomicalObject`s as input and generates new
    :class:`AstronomicalObject`s with the following transformations applied:
    - Drop random observations.
    - Drop large blocks of observations.
    - For galactic observations, adjust the brightness (= distance).
    - For extragalactic observations, adjust the redshift.
    - Add noise.

    The augmentor needs to have some reasonable idea of the properties of the
    survey that it is being applied to. If there is a large dataset that the
    classifier will be used on, then that dataset can be used directly to
    estimate the properties of the survey.

    This class needs to be subclassed to implement survey specific methods.
    These methods are:
    - TODO
    """
    def __init__(self):
        pass

    def augment_metadata(self, reference_object):
        """Generate new metadata for the augmented object.

        This method needs to be implemented in survey-specific subclasses of
        this class.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        Returns
        =======
        augmented_metadata : dict
            The augmented metadata
        """
        return NotImplementedError

    def augment_object(self, reference_object):
        """Generate an augmented version of an object.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        Returns
        =======
        augmented_object : :class:`AstronomicalObject`
            The augmented object.
        """
        # Create a new object id for the augmented object. We choose a random
        # string to add on to the end of the original object id that is very
        # unlikely to have collisions.
        ref_object_id = reference_object.metadata['object_id']
        random_str = ''.join(np.random.choice(list(string.ascii_letters), 10))
        new_object_id = '%s_aug_%s' % (ref_object_id, random_str)

        # Augment the metadata. This is survey specific, so this must be
        # implemented in subclasses.
        augmented_metadata = self.augment_metadata(reference_object)
        augmented_metadata['object_id'] = new_object_id
        augmented_metadata['reference_object_id'] = ref_object_id

        # Test
        return augmented_metadata
