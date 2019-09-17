"""Instrument specific definitions.

This module is used to define properties of various instruments. This should
eventually be split out into some kind of configuration file setup.
"""

band_central_wavelengths = {
    'lsstu': 3671.,
    'lsstg': 4827.,
    'lsstr': 6223.,
    'lssti': 7546.,
    'lsstz': 8691.,
    'lssty': 9710.,
}

# Colors for plotting
band_plot_colors = {
    'lsstu': 'C6',
    'lsstg': 'C4',
    'lsstr': 'C0',
    'lssti': 'C2',
    'lsstz': 'C3',
    'lssty': 'goldenrod',
}

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
    hasher.update(band.encode('utf8'))
    hex_color = '#%s' % hasher.hexdigest()[-6:]

    band_plot_colors[band] = hex_color

    return hex_color
