# Avocado

Photometric Classification of Astronomical Transients and Variables With Biased
Spectroscopic Samples

[![Documentation Status](https://readthedocs.org/projects/avocado-classifier/badge/?version=latest)](https://avocado-classifier.readthedocs.io/en/latest/?badge=latest)

## About

Avocado is a general photometric classification code that is designed to
produce classifications of arbitrary astronomical transients and variable
objects. This code is designed from the ground up to address the problem of
biased spectroscopic samples. It does so by generating many lightcurves from
each object in the original spectroscopic sample at a variety of redshifts and
with many different observing conditions. The "augmented" samples of
lightcurves that are generated are much more representative of the full
datasets than the original spectroscopic samples.

The original codebase of avocado was developed for and won the
[2018 Kaggle PLAsTiCC challenge](https://kaggle.com/c/PLAsTiCC-2018). A 
description of the algorithms used for this challenge can be found on the
[kaggle forum](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033). This
analysis can be replicated using the latest version of avocado following the
steps in the documentation.

## Installation

Instructions on how to install and use avocado can be found on the [avocado
readthedocs page](https://avocado-classifier.readthedocs.io/en/latest/).
