"""Instrument specific definitions.

This module is used to define properties of various instruments. This should
eventually be split out into some kind of configuration file setup.
"""
from .utils import AvocadoException

# Central wavelengths for each band.
band_central_wavelengths = {
    "lsstu": 3671.0,
    "lsstg": 4827.0,
    "lsstr": 6223.0,
    "lssti": 7546.0,
    "lsstz": 8691.0,
    "lssty": 9710.0,

    "ps1::g": 4866.0,
    "ps1::r": 6214.6,
    "ps1::i": 7544.6,
    "ps1::z": 8679.5,

    "desg": 4686.,
    "desr": 6166.,
    "desi": 7480.,
    "desz": 8932.,

    "ztfg": 4813.9,
    "ztfr": 6421.8,
    "ztfi": 7883.0,

    "uvot::u": 3475.5,
    "uvot::b": 4359.0,
    "uvot::v": 5430.1,
    "uvot::uvm2": 2254.9,
    "uvot::uvw1": 2614.1,
    "uvot::uvw2": 2079.0,
}

# Colors for plotting
band_plot_colors = {
    "lsstu": "C6",
    "lsstg": "C4",
    "lsstr": "C0",
    "lssti": "C2",
    "lsstz": "C3",
    "lssty": "goldenrod",

    "ps1::g": "C0",
    "ps1::r": "C2",
    "ps1::i": "C1",
    "ps1::z": "C3",

    "desg": "C4",
    "desr": "C0",
    "desi": "C2",
    "desz": "C3",

    "ztfg": "C0",
    "ztfr": "C2",
    "ztfi": "C1",
}

# Markers for plotting
band_plot_markers = {
    "lsstu": "o",
    "lsstg": "v",
    "lsstr": "^",
    "lssti": "<",
    "lsstz": ">",
    "lssty": "s",

    "ps1::g": "o",
    "ps1::r": "^",
    "ps1::i": "v",
    "ps1::z": "p",

    "desg": "v",
    "desr": "^",
    "desi": "<",
    "desz": ">",

    "ztfg": "o",
    "ztfr": "^",
    "ztfi": "v",
}


def get_band_central_wavelength(band):
    """Return the central wavelength for a given band.

    If the band does not yet have a color assigned to it, an AvocadoException
    is raised.

    Parameters
    ----------
    band : str
        The name of the band to use.
    """
    if band in band_central_wavelengths:
        return band_central_wavelengths[band]
    else:
        raise AvocadoException(
            "Central wavelength unknown for band %s. Add it to "
            "avocado.instruments.band_central_wavelengths." % band
        )


def get_band_plot_color(band):
    """Return the plot color for a given band.

    If the band does not yet have a color assigned to it, then a random color
    will be assigned (in a systematic way).

    Parameters
    ----------
    band : str
        The name of the band to use.
    """
    if band in band_plot_colors:
        return band_plot_colors[band]

    print("No plot color assigned for band %s, assigning a random one." % band)

    # Systematic random colors. We use the hash of the band name.
    # Note: hash() uses a random offset in python 3 so it isn't consistent
    # between runs!
    import hashlib

    hasher = hashlib.md5()
    hasher.update(band.encode("utf8"))
    hex_color = "#%s" % hasher.hexdigest()[-6:]

    band_plot_colors[band] = hex_color

    return hex_color


def get_band_plot_marker(band):
    """Return the plot marker for a given band.

    If the band does not yet have a marker assigned to it, then we use the
    default circle.

    Parameters
    ----------
    band : str
        The name of the band to use.
    """
    if band in band_plot_markers:
        return band_plot_markers[band]
    else:
        return "o"
