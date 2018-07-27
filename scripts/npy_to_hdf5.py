#!/usr/bin/env python

import argparse

import h5py

import numpy as np

usage_str = 'usage: %(prog)s [options] input.npy output.h5'
#description_str = 'pick lines from input and store in output'
description_str = 'convert npy to hdf5'

parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input_file', action='store', type=str, help='input.npy')
parser.add_argument('output_file', action='store', type=str, help='output.h5')
parser.add_argument('--key', action='store', dest='key', type=str, help='h5 key', default='data')
parser.add_argument('--gzip', action='store_true', dest='gzip', help='compress', default=False)
#parser.add_argument('--dtype', action='store', dest='dtype', type=str, help='dtype', default='float32')

args = parser.parse_args()

data = np.load(args.input_file)


with h5py.File(args.output_file, 'w') as f:
    if args.gzip:
        f.create_dataset(args.key, data=data.astype(data.dtype), compression='gzip')
    else:
        f.create_dataset(args.key, data=data.astype(data.dtype))
