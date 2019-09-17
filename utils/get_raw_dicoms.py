#!/usr/bin/env python

'''
get_raw_dicoms.py

get paths to raw dicom files for specific patients

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2019/02/12
'''


import sys

print('------')
print(' '.join(sys.argv))
print('------\n\n\n')

import tempfile
import os
import datetime
import time
import random
from warnings import warn
import configargparse as argparse

import numpy as np

import subtle.subtle_io as suio
import subtle.subtle_args as sargs
from distutils.dir_util import copy_tree


usage_str = 'usage: %(prog)s [options]'
description_str = 'Get raw dicom files'

if __name__ == '__main__':


    parser = sargs.parser(usage_str, description_str)
    args = parser.parse_args()

    args.path_zero = None
    args.path_low = None
    args.path_full = None

    args.path_zero, args.path_low, args.path_full = suio.get_dicom_dirs(args.path_base, override=args.override)

    _, path_zero = os.path.split(args.path_zero)
    _, path_low = os.path.split(args.path_low)
    _, path_full = os.path.split(args.path_full)

    for dicom_dir in suio.get_dicom_dirs(args.path_base, override=args.override):
        _, base_dir = os.path.split(dicom_dir)
        try:
            mydir = '{}/{}'.format(args.path_out, base_dir)
            os.mkdir(mydir)
            copy_tree(dicom_dir, mydir)
        except:
            pass
