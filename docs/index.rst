.. avocado documentation master file, created by
   sphinx-quickstart on Mon Apr 22 14:41:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Avocado
=======

*Photometric Classification of Astronomical Transients and Variables With
Biased Spectroscopic Samples*

About
-----

Avocado is a general photometric classification code that is designed to
produce classifications of arbitrary astronomical transients and variable
objects. This code is designed from the ground up to address the problem of
biased spectroscopic samples. It does so by generating many lightcurves from
each object in the original spectroscopic sample at a variety of redshifts and
with many different observing conditions. The "augmented" samples of
lightcurves that are generated are much more representative of the full
datasets than the original spectroscopic samples.

.. _installation:

Installation
------------

Requirements
............

Avocado has been tested on Python 3.7. It is not compatible with Python 2.
Avocado depends on the following packages:

- astropy
- george (available through the conda-forge channel on conda)
- lightgbm
- matplotlib
- numpy
- pandas
- pytables (pytables on conda, tables on pip)
- requests
- scikit-learn
- scipy
- tqdm

We recommend using `anaconda <https://www.anaconda.com/distribution/>`_ to
set up a python environment with these packages and using the
`conda-forge <https://conda-forge.org/>`_ channel.

Installation
............

Avocado can be downloaded from github using the following command: ::

    git clone https://github.com/kboone/avocado.git

It can then be installed as follows: ::

    cd avocado
    python setup.py install

Along with installing the avocado module, this package also provides a set of
scripts that can be used to process datasets on the command line for a variety
of common tasks.

Usage
-----

Avocado is designed to be a general purpose photometric classification code
that can be used for different surveys with implementations of different
classifiers. An example of how avocado can be applied to the PLAsTiCC dataset
using a LightGBM classifier can be seen :ref:`here <plasticc>`.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   plasticc


.. toctree::
   :maxdepth: 2

   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
