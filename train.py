'''
train.py

Training for contrast synthesis.
Trains netowrk using dataset of npy files

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/25
'''

import sys

import numpy as np
import os

import datetime
import time

import subtle.subtle_gad_network as sugn

import argparse

usage_str = 'usage: %(prog)s [options]'
description_str = 'train SubtleGrad network on pre-processed data'

# FIXME: add time stamps, logging
# FIXME: finish

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', action='store', dest='data_dir', type=str, help='directory containing pre-processed npy files', default=None)
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')

    args = parser.parse_args()
    
    verbose = args.verbose
    data_dir = args.data_dir

    assert data_dir is not None, 'must specify data directory'

    m = sugn.DeepEncoderDecoder2D(
            num_channel_input=2, num_channel_output=1,
            img_rows=512, img_cols=512,
            num_channel_first=32,
            verbose=verbose)

    m.model.fit(X, Y, batch_size=batch_size, epochs=epochs) 

