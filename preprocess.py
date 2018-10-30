#!/usr/bin/env python
'''
preprocess.py

Pre-processing for contrast synthesis.
Processes a set of dicoms and outputs h5 file
with scale/registration factors

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
    parser.add_argument('--normalize', action='store_true', dest='normalize', help="global scaling", default=False)
    parser.add_argument('--scale_matching', action='store_true', dest='scale_matching', help="match scaling of each image to each other", default=False)
    parser.add_argument('--joint_normalize', action='store_true', dest='joint_normalize', help="use same global scaling for all images", default=False)
    parser.add_argument('--normalize_fun', action='store', dest='normalize_fun', type=str, help='normalization fun', default='mean')
    parser.add_argument('--skip_registration', action='store_true', dest='skip_registration', help='skip co-registration', default=False)
    parser.add_argument('--skip_mask', action='store_true', dest='skip_mask', help='skip mask', default=False)
    parser.add_argument('--skip_scale_im', action='store_true', dest='skip_scale_im', help='skip histogram matching', default=False)

    args = parser.parse_args()

    metadata = {}

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

    normalize = args.normalize

    if args.normalize_fun == 'mean':
        normalize_fun = np.mean
    elif args.normalize_fun == 'max':
        normalize_fun = np.max
    else:
        assert 0, 'unrecognized normalization fun: {}'.format(args.normalize_fun)

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

    ims_zero, hdr_zero = suio.dicom_files(path_zero, normalize=False)
    ims_low, hdr_low = suio.dicom_files(path_low, normalize=False)
    ims_full, hdr_full = suio.dicom_files(path_full, normalize=False)

    pixel_spacing_zero = suio.get_pixel_spacing(hdr_zero)
    pixel_spacing_low = suio.get_pixel_spacing(hdr_low)
    pixel_spacing_full = suio.get_pixel_spacing(hdr_full)

    metadata['pixel_spacing_zero'] = pixel_spacing_zero
    metadata['pixel_spacing_low'] = pixel_spacing_low
    metadata['pixel_spacing_full'] = pixel_spacing_full

    if verbose:
        print('image sizes: ', ims_zero.shape, ims_low.shape, ims_full.shape)

    # FIXME: assert that number of slices are the same
    ns, nx, ny = ims_zero.shape

    idx_start = int(ns * discard_start_percent) # inclusive
    idx_end = int(ns * (1 - discard_end_percent)) # not inclusive
    idx = np.arange(idx_start, idx_end)

    metadata['slice_idx'] = idx
    
    if verbose and idx_start > 0:
        if verbose:
            print('discarding first {:d} slices'.format(idx_start))

    if verbose and idx_end < ns - 1:
        print('discarding last {:d} slices'.format(ns - idx_end))

    ims = np.stack((ims_zero[idx,:,:], ims_low[idx,:,:], ims_full[idx,:,:]), axis=1)

    ns, nc, nx, ny = ims.shape

    if verbose:
        print('masking')

    ### MASKING ###
    if not args.skip_mask:
        mask = sup.mask_im(ims, threshold=mask_threshold)
        metadata['mask'] = 1
        metadata['mask_threshold'] = mask_threshold
    else:
        mask = np.ones(ims.shape)
        metadata['mask'] = 0

    ims *= mask


    ### HISTOGRAM NORMALIZATION ###
    if not args.skip_scale_im:
        metadata['hist_norm'] = 1
        # FIXME: expose to outside world. subject to change once we implement white striping
        levels=1024
        points=50
        mean_intensity=True

        ims[:,1,:,:] = sup.scale_im(ims[:,0,:,:], ims[:,1,:,:], levels, points, mean_intensity)
        ims[:,2,:,:] = sup.scale_im(ims[:,0,:,:], ims[:,2,:,], levels, points, mean_intensity)
    else:
        metadata['hist_norm'] = 0


    ### IMAGE REGISTRATION ###
    spars = sitk.GetDefaultParameterMap(transform_type)

    if not args.skip_registration:
        metadata['reg'] = 1
        metadata['transform_type'] = transform_type
        ims[:,1,:,:], spars1_reg = sup.register_im(ims[:,0,:,:], ims[:,1,:,:], param_map=spars, verbose=verbose, im_fixed_spacing=pixel_spacing_zero, im_moving_spacing=pixel_spacing_low)

        if verbose:
            print('low dose transform parameters: {}'.format(spars1_reg[0]['TransformParameters']))

    if not args.skip_registration:
        ims[:,2,:,:], spars2_reg = sup.register_im(ims[:,0,:,:], ims[:,2,:,:], param_map=spars, verbose=verbose, im_fixed_spacing=pixel_spacing_zero, im_moving_spacing=pixel_spacing_full)

        if verbose:
            print('full dose transform parameters: {}'.format(spars2_reg[0]['TransformParameters']))
    else:
        metadata['reg'] = 0

    # for scaling
    nslices = 20
    idx_scale = range(ns//2 - nslices // 2, ns//2 + nslices // 2)

    m = mask[idx_scale, 0, :, :]
    im0, im1, im2 = ims[idx_scale, 0, :, :], ims[idx_scale, 1, :, :], ims[idx_scale, 2, :, :]

    im0 = im0[m != 0].ravel()
    im1 = im1[m != 0].ravel()
    im2 = im2[m != 0].ravel()

    _ims = np.stack((im0, im1, im2), axis=1)

    metadata['scale_slices'] = idx_scale

    ### IMAGE SCALE MATCHING ###
    if args.scale_matching:
        if verbose:
            print('intensity before scaling:')
            print('mean', np.mean(np.abs(_ims), axis=(0)))
            print('median', np.median(np.abs(_ims), axis=(0)))
            print('max', np.max(np.abs(_ims), axis=(0)))

        levels = np.linspace(.5, 1.5, 30)
        max_iter = 3

        ntic = time.time()
        scale_low = sup.scale_im_enhao(im0, im1, levels=levels, max_iter=max_iter)
        scale_full = sup.scale_im_enhao(im0, im2, levels=levels, max_iter=max_iter)

        metadata['scale_low'] = scale_low
        metadata['scale_full'] = scale_full

        ntoc = time.time()

        if verbose:
            print('scale low:', scale_low)
            print('scale full:', scale_full)
            print('done scaling data ({:.2f} s)'.format(ntoc - ntic))

        ims[:,1,:,:] = ims[:,1,:,:] * scale_low
        ims[:,2,:,:] = ims[:,2,:,:] * scale_full

        _ims[:,1] = _ims[:,1] * scale_low
        _ims[:,2] = _ims[:,2] * scale_full

        if verbose:
            print('intensity after scaling:')
            print('mean', np.mean(np.abs(_ims), axis=(0)))
            print('median', np.median(np.abs(_ims), axis=(0)))
            print('max', np.max(np.abs(_ims), axis=(0)))


    ### GLOBAL NORMALIZATION ###
    if args.normalize:
        if verbose:
            print('normalizing with function ', args.normalize_fun, normalize_fun)

        if args.joint_normalize:
            axis=(0,1)
        else:
            axis=(0)

        ntic = time.time()

        scale_global = sup.normalize_scale(_ims, axis=axis, fun=normalize_fun)
        metadata['scale_global'] = scale_global

        if verbose:
            ntoc = time.time()
            print('global scaling:', scale_global)
            print('done ({:.2f}s)'.format(ntoc - ntic))

        ims = ims / scale_global[:,:,None,None]
        _ims = _ims / scale_global


    suio.save_data_h5(out_file, data=ims, h5_key='data', metadata=metadata)
