import logging
import os

from . import settings


# Logging
logger = logging.getLogger('avocado')


# Exceptions
class AvocadoException(Exception):
    pass


def write_dataframes(path, dataframes, keys, overwrite=False, append=False,
                     timeout=5):
    """Write a set of dataframes out to an HDF5 file

    The append functionality is designed so that multiple independent processes
    running simultaneously can append to the same file. Each process will lock
    the output file while it is writing, and other processes will repeatedly
    try to get the lock until they succeed. With this implementation, if the
    file is locked by other means, the processes will hang endlessly until the
    lock is released.

    Parameters
    ----------
    path : str
        The output file path
    dataframes : list
        A list of pandas DataFrame objects to write out.
    keys : list
        A list of keys to use in the HDF5 file for each DataFrame.
    overwrite : bool
        If there is an existing file at the given path, it will be deleted if
        overwrite is True. Otherwise an exception will be raised.
    append : bool
        If True, the dataframes will be appended to the file if a file exists
        at the given path.
    timeout : int
        After failing to write to a file in append mode, wait this amount of
        time in seconds before retrying the write (to allow other processes to
        finish).
    """
    from tables.exceptions import HDF5ExtError
    import time

    # Make the containing directory if it doesn't exist yet.
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    # Handle if the file already exists.
    if os.path.exists(path):
        if overwrite:
            logger.warning("Overwriting %s..." % path)
            os.remove(path)
        elif append:
            # We are appending to the file, so it is fine to have a file there
            # already.
            pass
        else:
            raise AvocadoException(
                "Dataset %s already exists! Can't write." % path
            )

    for dataframe, key in zip(dataframes, keys):
        while True:
            # When appending, we repeatedly try to write to the file so that
            # many processes can write to the same file at the same time.
            try:
                dataframe.to_hdf(path, key, mode='a', append=append,
                                 format='table', data_columns=['object_id'])
            except HDF5ExtError:
                # Failed to write the file, try again if we are in append mode
                # (otherwise this shouldn't happen).
                if not append:
                    raise

                timeout = 5

                logger.warning(
                    "Error writing to HDF5 file %s... another process is "
                    "probably using it. Retrying in %d seconds."
                    % (path, timeout)
                )
                time.sleep(timeout)
            else:
                break
