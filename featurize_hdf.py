#!/usr/bin/env python

import argparse
import pandas as pd

import plasticc_gp

# parser = argparse.ArgumentParser()
# parser.add_argument('id', type=int)
# args = parser.parse_args()

basedir = '/home/scpdata06/kboone/plasticc/'

dataset = plasticc_gp.Dataset()
dataset.load_augment(40)
dataset.extract_all_features()
