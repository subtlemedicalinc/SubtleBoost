#!/usr/bin/env python
'''
write_dicoms.py

Writes a predicted image volume to dicoms

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/10/30
'''

import sys

import numpy as np
import os

import datetime
import time

import subtle.subtle_preprocess as supre
import subtle.utils.io as utils_io

import argparse

usage_str = 'usage: %(prog)s [options]'
description_str = 'write predicted volume to dicoms'

# FIXME: add time stamps, logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_ref', action='store', dest='path_ref', type=str, help='path to reference dicom dir', default=None)
    parser.add_argument('--path_out', action='store', dest='path_out', type=str, help='path to output SubtleGad dicom dir', default=None)
    parser.add_argument('--input', action='store', dest='input', type=str, help='path to input data file (h5 or npy)', default=None)
    parser.add_argument('--metadata', action='store', dest='metadata', type=str, help='path to metadata h5 file if different from input)', default=None)
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')
    parser.add_argument('--description', action='store', dest='description', type=str, help='append to end of series description', default='')

    args = parser.parse_args()

    data_in = utils_io.load_file(args.input)

    if args.metadata:
        metadata = utils_io.load_h5_metadata(args.metadata)
    else:
        metadata = utils_io.load_h5_metadata(args.input)

    data_out = supre.undo_scaling(data_in, metadata, verbose=args.verbose)

    utils_io.write_dicoms(args.path_ref, data_out, args.path_out, series_desc_pre='SubtleGad: ', series_desc_post=args.description)
