#!/usr/bin/env python

import argparse
import pathlib

import  subtle.utils.io as utils_io

usage_str = 'usage: %(prog)s [options] <input> <output>'
#description_str = 'pick lines from input and store in output'
description_str = 'convert file type 1 to file type 2'

parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input_file', action='store', type=str, help='input')
parser.add_argument('output_file', action='store', type=str, help='output')
parser.add_argument('--key', action='store', dest='key', type=str, help='h5 key', default='data')

args = parser.parse_args()

# infer data type from file extensions

data = utils_io.load_file(args.input_file, params={'h5_key': args.key})

suffix = ''.join(pathlib.Path(args.output_file).suffixes)

if suffix == '.h5z':
    utils_io.save_data_h5(args.output_file, data, h5_key=args.key, compress=True)
else:
    utils_io.save_data(args.output_file, data, params={'h5_key': args.key, 'compress': False})
