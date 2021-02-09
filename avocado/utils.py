import logging
import os
import pandas as pd
import requests
import sys
from tqdm import tqdm
from urllib.request import urlretrieve


# Logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("avocado")


# Exceptions
class AvocadoException(Exception):
    """The base class for all exceptions raised in avocado."""

    pass


def _verify_hdf_chunks(store, keys):
    """Verify that a pandas table that was written to an HDF5 file in chunks
    was fully written out.

    If successful, this will return normally. Otherwise, it will raise an
    exception indicating which chunks missing in the file.

    Parameters
    ----------
    store : `pandas.HDFStore`
        The HDF5 file to verify.
    keys : list
        A list of keys to verify in the HDF5 file.
    """
    if "chunk_info" not in store:
        # No chunk_info. The file wasn't written by chunks so there is nothing
        # to do.
        return

    chunk_info = pd.read_hdf(store, "chunk_info")

    missing_chunks = {key: list() for key in keys}

    valid = True

    for chunk_id, chunk_str in chunk_info.itertuples():
        chunk_keys = chunk_str.split(",")

        for key in keys:
            if key not in chunk_keys:
                missing_chunks[key].append(chunk_id)
                valid = False

    if not valid:
        # Missing some keys, raise an exception.
        missing_str = ""
        show_chunks = 5
        for key, chunks in missing_chunks.items():
            if len(chunks) == 0:
                continue

            chunk_str = ", ".join([str(i) for i in chunks[:5]])
            if len(chunks) > show_chunks:
                chunk_str += ", ... (%d total)" % len(chunks)
            missing_str += "\n    %s: %s" % (key, chunk_str)

        message = "File %s is missing the following chunks: %s" % (
            store.filename,
            missing_str,
        )

        raise AvocadoException(message)


def _map_column_name(key_store, target_name):
    """Figure out the internal column name for a given target name

    If a column is used as the index of a table, its PyTables name is 'index'
    instead of the desired name. Handle that gracefully.
    """
    try:
        index_name = key_store.attrs.info["index"]["index_name"]
    except KeyError:
        index_name = ""

    if index_name == target_name:
        return "index"
    else:
        return target_name


def read_dataframes(
    path,
    keys,
    chunk=None,
    num_chunks=None,
    chunk_column="object_id",
    verify_input_chunks=True,
):
    """Read a set of pandas DataFrames from an HDF5 file

    Optionally, the DataFrames can be read in chunks. If that is the case, the
    column specified by `chunk_column` will be loaded from the DataFrame
    corresponding to the first key in keys. This column will be sorted, and
    split into `num_chunks` equal length chunks. Only the data from the chunk
    specified by `chunk` will be loaded. For all other keys, only rows with a
    matching value of `chunk_column` to the ones selected for the chunk will be
    loaded.

    When reading in chunks, all DataFrames must have the key 'chunk_column'
    available. The first DataFrame must have that column indexed.

    Files can also be written by chunks with `write_dataframe`. When loading
    such a file, we check to make sure that the data for all chunks was
    successfully written, and we raise an exception if chunks are missing. This
    behavior can be overruled by setting `verify_input_chunks` to False.

    Parameters
    ----------
    path : str
        The input file path
    keys : list
        A list of keys to read each DataFrame from in the HDF5 file.
    chunk : int (optional)
        If set, the dataset will be loaded in chunks. This specifies the chunk
        number.
    num_chunks : int (optional)
        If loading the dataset in chunks, this is the total number of chunks to
        use.
    chunk_column : str (optional)
        The column to use for separating chunks.
    verify_input_chunks : bool (optional)
        If True, and the input file was written in chunks, this will verify
        that all chunks are available for the input file.

    Returns
    -------
    dataframes : list
        The list of :class:`pandas.DataFrame` objects that were read.
    """
    store = pd.HDFStore(path, "r")

    # If the input was written in chunks, verify that they are all available.
    if verify_input_chunks:
        _verify_hdf_chunks(store, keys)

    if chunk is None:
        # Not loading in chunks, just load all of the datasets normally.
        dataframes = []
        for key in keys:
            dataframe = pd.read_hdf(store, key)
            dataframe.sort_index(inplace=True)
            dataframes.append(dataframe)
        store.close()
        return dataframes

    # Load a chunk of the dataset.

    if num_chunks is None:
        store.close()
        raise AvocadoException(
            "num_chunks must be specified to load the data in chunks!"
        )

    if chunk < 0 or chunk >= num_chunks:
        store.close()
        raise AvocadoException("chunk must be in range [0, num_chunks)!")

    first = True

    dataframes = []

    for key in keys:
        key_store = store.get_storer(key)
        use_name = _map_column_name(key_store, chunk_column)

        if first:
            # Use the first DataFrame to figure out the range of values of the
            # chunk column that we want to load.

            # Need to make sure that there is a CSI index for this column (one
            # that is fully sorted). If not, we can't use these tricks.
            # Calling `write_dataframe` with index_chunk_column=True ensures
            # that the chunk_column has a CSI index.
            index = key_store.table.colindexes[use_name]
            if not index.is_csi:
                raise AvocadoException(
                    "Error: can only do chunking on columns with a CSI index!"
                )

            # Use inclusive start and end indices and figure out the rows at
            # those boundaries.
            num_rows = index.nelements
            start_idx = chunk * num_rows // num_chunks
            end_idx = (chunk + 1) * num_rows // num_chunks - 1

            start_object_id = index.read_sorted(start_idx, start_idx + 1)[0]
            end_object_id = index.read_sorted(end_idx, end_idx + 1)[0]

            start_object_id = start_object_id.decode().strip()
            end_object_id = end_object_id.decode().strip()

            first = False

        # Read the DataFrame
        match_str = "(%s >= '%s') & (%s <= '%s')" % (
            use_name,
            start_object_id,
            use_name,
            end_object_id,
        )
        dataframe = pd.read_hdf(store, key, where=match_str)
        dataframe.sort_index(inplace=True)

        dataframes.append(dataframe)

    store.close()
    return dataframes


def read_dataframes_query(path, keys, query_key, query_column='object_id'):
    """Read a set of pandas DataFrames for rows in an HDF5 matching a given query.

    This is primarily used to read the observations for a single object without
    having to load all of the other objects. To do this, pass the object ID as
    `query_key`.

    Parameters
    ----------
    path : str
        Input file path
    keys : list
        A list of keys to read each DataFrame from in the HDF5 file.
    query_key : str
        Key to query for.
    query_column : str
        Column that the key is in.

    Returns
    -------
    dataframes : list
        The list of :class:`pandas.DataFrame` objects that were read.
    """
    store = pd.HDFStore(path, "r")

    dataframes = []

    for key in keys:
        key_store = store.get_storer(key)
        use_query_column = _map_column_name(key_store, query_column)
        match_str = f"{use_query_column} == {query_key}"
        dataframe = pd.read_hdf(store, key, where=match_str)
        dataframe.sort_index(inplace=True)
        dataframes.append(dataframe)

    store.close()
    return dataframes


def write_dataframe(
    path,
    dataframe,
    key,
    overwrite=False,
    append=None,
    timeout=5,
    chunk=None,
    num_chunks=None,
    chunk_column="object_id",
    index_chunk_column=True,
):
    """Write a dataframe out to an HDF5 file

    The append functionality is designed so that multiple independent
    processes, potentially running simultaneously, can append to the same file.
    Each process will lock the output file while it is writing, and other
    processes will repeatedly try to get the lock until they succeed. With this
    implementation, if the file is locked by other means, the processes will
    hang endlessly until the lock is released.

    Typically, the append functionality is used when the writing process is
    operating on an input file with multiple chunks. If "chunk" and
    "num_chunks" are passed, then this writer keeps track of which chunks have
    been processed and which haven't. `read_dataframes` will then be able to
    tell if it is reading a complete file or not.

    Parameters
    ----------
    path : str
        The output file path.
    dataframe : :class:`pandas.DataFrame`
        The pandas DataFrame object to write out.
    key : str
        The key to write the DataFrame out to.
    overwrite : bool
        If there is an existing file at the given path, it will be deleted if
        overwrite is True. Otherwise an exception will be raised. overwrite is
        not supported if using the append functionality.
    append : bool
        If True, the dataframe will be appended to the file if a file exists at
        the given path. defualt: True if chunk is set, False otherwise.
    timeout : int
        After failing to write to a file in append mode, wait this amount of
        time in seconds before retrying the write (to allow other processes to
        finish).
    chunk : int (optional)
        If the dataset was loaded in chunks, this indicates the chunk number.
    num_chunks : int (optional)
        If the dataset was loaded in chunks, this is the total number of chunks
        used.
    chunk_column : str (optional)
        If the dataset was loaded in chunks, this is the column that was used
        for separating chunks.
    index_chunk_column : bool (optional)
        If True (default), create a PyTables CSI index on the chunk column.
        This is necessary for us to be able to choose chunks of the dataset
        without having to read everything at once. The index can be created
        later with `_create_csi_index` after the DataFrame has fully been
        written to disk if necessary.
    """
    from tables.exceptions import HDF5ExtError
    import time

    if append is None:
        if chunk is not None:
            append = True
        else:
            append = False

    # Make the containing directory if it doesn't exist yet.
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    # Handle if the file already exists.
    if os.path.exists(path):
        if overwrite:
            logger.warning("Overwriting %s..." % path)
            os.remove(path)
        elif append:
            # We are appending to the file, so it is fine to have a file
            # there already.
            pass
        else:
            raise AvocadoException("Dataset %s already exists! Can't write." % path)

    # Get a lock on the HDF5 file.
    while True:
        try:
            store = pd.HDFStore(path, "a")
        except HDF5ExtError:
            # Failed to get a lock on the file. If we are in append mode,
            # another process likely has the lock already, so wait and try
            # again. If we aren't in append mode, this shouldn't happen.
            if not append:
                raise

            logger.info(
                "Couldn't get lock for HDF5 file %s... another process is "
                "probably using it. Retrying in %d seconds." % (path, timeout)
            )

            time.sleep(timeout)
        else:
            # Got the lock successfully.
            break

    # If we are writing out chunks, make sure that this chunk hasn't already
    # been written to this file.
    if chunk is not None:
        if "chunk_info" in store:
            chunk_info = pd.read_hdf(store, "chunk_info")
        else:
            # No chunk_info yet, create it.
            chunk_index = [i for i in range(num_chunks)]
            chunk_values = ["" for i in range(num_chunks)]
            chunk_info = pd.DataFrame({"keys": chunk_values}, index=chunk_index)

        # Make sure that the chunk_info has the right number of chunks.
        file_num_chunks = len(chunk_info)
        if file_num_chunks != num_chunks:
            raise AvocadoException(
                "Mismatched number of chunks (current %d, requested %d) "
                "for file %s." % (file_num_chunks, num_chunks, path)
            )

        # Make sure that we haven't already written any of the keys.
        written_keys = chunk_info.loc[chunk, "keys"].split(",")

        if key in written_keys:
            raise AvocadoException(
                "Error writing %s! Already found key %s for chunk %d!"
                % (path, key, chunk)
            )

        written_keys.append(key)

        # Remove the empty key that is created the first time around.
        if "" in written_keys:
            written_keys.remove("")

        # Update the written keys with the ones that we are adding.
        chunk_info.loc[chunk, "keys"] = ",".join(written_keys)
        chunk_info.to_hdf(store, "chunk_info", format="table")

    dataframe.to_hdf(
        store, key, mode="a", append=append, format="table", data_columns=[chunk_column]
    )

    if index_chunk_column:
        _create_csi_index(store, key, chunk_column)

    store.close()


def _create_csi_index(store, key, column_name):
    """Create a CSI index on a column in an HDF5 file.

    The column must have been already specified in the data_columns call to
    to_hdf or it won't be stored correctly in the HDF5 file.

    Parameters
    ----------
    store : :class:`pandas.HDFStore`
        An HDF5 file opened as an instance of a :class:`pandas.HDFStore`
        object.
    key : str
        The key of the DataFrame to use.
    column_name : str
        The column to add a CSI index to.
    """
    key_store = store.get_storer(key)
    use_name = _map_column_name(key_store, column_name)
    column = key_store.table.colinstances[use_name]

    if not column.index.is_csi:
        column.remove_index()
        column.create_csindex()


def read_dataframe(path, key, *args, **kwargs):
    """Read a single DataFrame out to an HDF5 file

    This is just a thin wrapper around read_dataframes.
    """
    return read_dataframes(path, [key], **kwargs)[0]


def download_file(url, path, filesize=None):
    """Download a file with a tqdm progress bar. This code is adapted from an
    example in the tqdm documentation.
    """
    # Check if the file already exists, and don't download it if it does.
    if os.path.exists(path):
        # Make sure that we have a full download
        if filesize is None or os.path.getsize(path) == filesize:
            print("Skipping %s, already exists." % os.path.basename(path))
            return
        else:
            print("Found incomplete download of %s, retrying." %
                  os.path.basename(path))
            os.remove(path)

    class TqdmUpTo(tqdm):
        """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
        def update_to(self, b=1, bsize=1, tsize=None):
            """
            b  : int, optional
                Number of blocks transferred so far [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)  # will also set self.n = b * bsize

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                  desc=url.split('/')[-1]) as t:  # all optional kwargs
        urlretrieve(url, filename=path, reporthook=t.update_to, data=None)


def download_zenodo(record, basedir):
    # Make the download directory if it doesn't exist.
    os.makedirs(basedir, exist_ok=True)

    # Download a dataset from zenodo.
    zenodo_url = f"https://zenodo.org/api/records/{record}"
    zenodo_metadata = requests.get(zenodo_url).json()

    for file_metadata in zenodo_metadata['files']:
        path = os.path.join(basedir, file_metadata['key'])
        url = file_metadata['links']['self']
        filesize = file_metadata['size']

        download_file(url, path, filesize)
