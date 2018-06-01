#!/usr/bin/env python
'''
preprocess.py

Pre-processing for contrast synthesis.
Processes a set of dicoms and outputs npy file

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/25
'''

import sys

import numpy as np
import os

import datetime
import time

sys.path.insert(0, '/home/subtle/jon/tools/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.5/SimpleITK')
import SimpleITK as sitk

import subtle.subtle_preprocess as sup
import subtle.subtle_io as suio

import argparse

usage_str = 'usage: %(prog)s [options]'
description_str = 'pre-process data for SubtleGad project'

# FIXME: add time stamps, logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_zero', action='store', dest='path_zero', type=str, help='path to zero dose dicom dir', default=None)
    parser.add_argument('--path_low', action='store', dest='path_low', type=str, help='path to low dose dicom dir', default=None)
    parser.add_argument('--path_full', action='store', dest='path_full', type=str, help='path to full dose dicom dir', default=None)
    parser.add_argument('--path_base', action='store', dest='path_base', type=str, help='path to base dicom directory containing subdirs', default=None)
    parser.add_argument('--output', action='store', dest='out_file', type=str, help='output to npy file', default='out.npy')
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')
    parser.add_argument('--discard_start_percent', action='store', type=float, dest='discard_start_percent', help='throw away start X %% of slices', default=0.)
    parser.add_argument('--discard_end_percent', action='store', type=float, dest='discard_end_percent', help='throw away end X %% of slices', default=0.)
    parser.add_argument('--mask_threshold', action='store', type=float, dest='mask_threshold', help='cutoff threshold for mask', default=.08)
    parser.add_argument('--transform_type', action='store', type=str, dest='transform_type', help="transform type ('rigid', 'translation', etc.)", default='rigid')

    args = parser.parse_args()

    path_zero = args.path_zero
    path_low = args.path_low
    path_full = args.path_full
    path_base = args.path_base
    out_file = args.out_file
    
    verbose = args.verbose

    discard_start_percent = args.discard_start_percent
    discard_end_percent = args.discard_end_percent

    mask_threshold = args.mask_threshold
    transform_type = args.transform_type

    if path_zero is not None and path_low is not None and path_full is not None:
        use_indiv_path = True
    else:
        use_indiv_path = False

    if path_base is not None:
        use_base_path = True
        assert not use_indiv_path, 'cannot specify both base path and individual paths'
    else:
        use_base_path = False

    assert use_base_path or use_indiv_path, 'must specify base path or individual paths'

    assert discard_start_percent >= 0 and discard_start_percent < 1
    assert discard_end_percent >= 0 and discard_end_percent < 1


    if use_base_path:
        path_zero, path_low, path_full = suio.get_dicom_dirs(path_base)
        if verbose:
            print('path_zero = {}'.format(path_zero))
            print('path_low = {}'.format(path_low))
            print('path_full = {}'.format(path_full))

    ims_zero, hdr_zero = suio.dicom_files(path_zero)
    ims_low, hdr_low = suio.dicom_files(path_low)
    ims_full, hdr_full = suio.dicom_files(path_full)

    if verbose:
        print('image sizes: ', ims_zero.shape, ims_low.shape, ims_full.shape)

    # FIXME: assert that number of slices are the same
    ns, nx, ny = ims_zero.shape

    idx_start = int(ns * discard_start_percent) # inclusive
    idx_end = int(ns * (1 - discard_end_percent)) # not inclusive
    idx = np.arange(idx_start, idx_end)
    
    if verbose and idx_start > 0:
        if verbose:
            print('discarding first {:d} slices'.format(idx_start))

    if verbose and idx_end < ns - 1:
        print('discarding last {:d} slices'.format(ns - idx_end))

    ims = np.stack((ims_zero[idx,:,:], ims_low[idx,:,:], ims_full[idx,:,:]), axis=1)

    ns, nc, nx, ny = ims.shape

    if verbose:
        print('masking')

    mask = sup.mask_im(ims, threshold=mask_threshold)
    ims *= mask

    # FIXME: expose to outside world. subject to change once we implement white striping
    levels=1024
    points=50
    mean_intensity=True

    ims[:,1,:,:] = sup.scale_im(ims[:,0,:,:], ims[:,1,:,:], levels, points, mean_intensity)
    ims[:,2,:,:] = sup.scale_im(ims[:,0,:,:], ims[:,2,:,], levels, points, mean_intensity)

    spars = sitk.GetDefaultParameterMap(transform_type)

    ims[:,1,:,:], spars1_reg = sup.register_im(ims[:,0,:,:], ims[:,1,:,:], param_map=spars, verbose=verbose)

    if verbose:
        print('low dose transform parameters: {}'.format(spars1_reg[0]['TransformParameters']))

    ims[:,2,:,:], spars2_reg = sup.register_im(ims[:,0,:,:], ims[:,2,:,:], param_map=spars, verbose=verbose)

    if verbose:
        print('full dose transform parameters: {}'.format(spars2_reg[0]['TransformParameters']))

    np.save(out_file, ims)
