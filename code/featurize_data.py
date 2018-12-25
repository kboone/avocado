#!/usr/bin/env python

import argparse

import plasticc_gp

parser = argparse.ArgumentParser()
parser.add_argument('chunk', type=int)
args = parser.parse_args()

basedir = '/home/scpdata06/kboone/plasticc/'

dataset = plasticc_gp.Dataset()
dataset.load_chunk(args.chunk)

print("Found %d targets for chunk %d" % (len(dataset.meta_data), args.chunk))

dataset.extract_all_features()
