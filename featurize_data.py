#!/usr/bin/env python

import argparse
import pandas as pd

import plasticc_gp

parser = argparse.ArgumentParser()
parser.add_argument('chunk', type=int)
args = parser.parse_args()

basedir = '/home/scpdata06/kboone/plasticc/'

dataset = plasticc_gp.Dataset()
dataset.load_chunk(args.chunk)

print("Found %d targets for chunk %d" % (len(dataset.meta_data), args.chunk))

all_features = []
for i in range(len(dataset.meta_data)):
    print(i)
    feature_labels, features = dataset.extract_features(i)
    all_features.append(features)

feature_table = pd.DataFrame(all_features, columns=feature_labels)

feature_table.to_hdf(
    '/home/scpdata06/kboone/plasticc/features/features_%04d.h5' % args.chunk,
    'features', mode='w'
)
