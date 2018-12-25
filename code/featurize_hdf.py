#!/usr/bin/env python

import argparse
import pandas as pd

import plasticc_gp

dataset = plasticc_gp.Dataset()
dataset.load_augment(40)
dataset.extract_all_features()
