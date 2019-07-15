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

from scipy.ndimage import zoom

sys.path.insert(0, '/home/subtle/jon/tools/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.5/SimpleITK')
import SimpleITK as sitk

import subtle.subtle_preprocess as sup
import subtle.subtle_io as suio

import argparse

def fetch_args():
    usage_str = 'usage: %(prog)s [options]'
    description_str = 'pre-process data for SubtleGad project'

    # FIXME: add time stamps, logging
    parser = argparse.ArgumentParser(
        usage=usage_str, description=description_str,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--path_zero', action='store', dest='path_zero',
                        type=str, help='path to zero dose dicom dir',
                        default=None)
    parser.add_argument('--path_low', action='store', dest='path_low',
                        type=str, help='path to low dose dicom dir',
                        default=None)
    parser.add_argument('--path_full', action='store', dest='path_full',
                        type=str, help='path to full dose dicom dir',
                        default=None)
    parser.add_argument('--path_base', action='store', dest='path_base',
                        type=str, help='path to base dicom directory containing subdirs', default=None)
    parser.add_argument('--output', action='store', dest='out_file', type=str,
                        help='output to npy file', default='out.npy')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        help='verbose')
    parser.add_argument('--discard_start_percent', action='store', type=float,
                        dest='discard_start_percent', help='throw away start X %% of slices', default=0.)
    parser.add_argument('--discard_end_percent', action='store', type=float,
                        dest='discard_end_percent', help='throw away end X %% of slices', default=0.)
    parser.add_argument('--mask_threshold', action='store', type=float,
                        dest='mask_threshold', help='cutoff threshold for mask', default=.08)
    parser.add_argument('--transform_type', action='store', type=str,
                        dest='transform_type', help="transform type ('rigid', 'translation', etc.)", default='rigid')
    parser.add_argument('--normalize', action='store_true', dest='normalize',
                        help="global scaling", default=False)
    parser.add_argument('--scale_matching', action='store_true',
                        dest='scale_matching', help="match scaling of each image to each other", default=False)
    parser.add_argument('--joint_normalize', action='store_true',
                        dest='joint_normalize', help="use same global scaling for all images", default=False)
    parser.add_argument('--global_scale_ref_im0', action='store_true',
                        dest='global_scale_ref_im0', help="use zero-dose for global scaling ref", default=False)
    parser.add_argument('--normalize_fun', action='store',
                        dest='normalize_fun', type=str, help='normalization fun', default='mean')
    parser.add_argument('--skip_registration', action='store_true',
                        dest='skip_registration', help='skip co-registration',
                        default=False)
    parser.add_argument('--skip_mask', action='store_true', dest='skip_mask',
                        help='skip mask', default=False)
    parser.add_argument('--skip_scale_im', action='store_true',
                        dest='skip_scale_im', help='skip histogram matching',
                        default=False)
    parser.add_argument('--override_dicom_naming', action='store_true',
                        dest='override', help='dont check dicom names',
                        default=False)
    parser.add_argument('--scale_dicom_tags', action='store_true',
                        dest='scale_dicom_tags', help='use dicom tags for relative scaling', default=False)
    parser.add_argument('--zoom', action='store', dest='zoom', type=int,
                        help='zoom to in-plane matrix size', default=None)
    parser.add_argument('--zoom_order', action='store', dest='zoom_order',
                        type=int, help='zoom order', default=3)
    parser.add_argument('--nslices', action='store', dest='nslices', type=int,
                        help='number of slices for scaling', default=20)

    args = parser.parse_args()
    return args

def _assert_and_get_init_vars(args):
    if args.normalize_fun == 'mean':
        normalize_fun = np.mean
    elif args.normalize_fun == 'max':
        normalize_fun = np.max
    else:
        assert 0, 'unrecognized normalization fun: {}'.format(args.normalize_fun)

    if (args.path_zero is not None and
        args.path_low is not None and
        args.path_full is not None
    ):
        use_indiv_path = True
    else:
        use_indiv_path = False

    if args.path_base is not None:
        use_base_path = True
        assert not use_indiv_path, 'cannot specify both base path and individual paths'
    else:
        use_base_path = False

    assert use_base_path or use_indiv_path, 'must specify base path or individual paths'

    assert args.discard_start_percent >= 0 and args.discard_start_percent < 1
    assert args.discard_end_percent >= 0 and args.discard_end_percent < 1

    return normalize_fun, use_indiv_path, use_base_path

def _get_images(args, metadata):
    normalize_fun, use_indiv_path, use_base_path = _assert_and_get_init_vars(args)

    metadata['normalize_fun'] = normalize_fun
    metadata['use_indiv_path'] = use_indiv_path
    metadata['use_base_path'] = use_base_path

    if use_base_path:
        args.path_zero, args.path_low, args.path_full = suio.get_dicom_dirs(args.path_base, override=args.override)
        if args.verbose:
            print('path_zero = {}'.format(args.path_zero))
            print('path_low = {}'.format(args.path_low))
            print('path_full = {}'.format(args.path_full))

    ims_zero, hdr_zero = suio.dicom_files(args.path_zero, normalize=False)
    ims_low, hdr_low = suio.dicom_files(args.path_low, normalize=False)
    ims_full, hdr_full = suio.dicom_files(args.path_full, normalize=False)

    pixel_spacing_zero = suio.get_pixel_spacing(hdr_zero)
    pixel_spacing_low = suio.get_pixel_spacing(hdr_low)
    pixel_spacing_full = suio.get_pixel_spacing(hdr_full)

    metadata['pixel_spacing_zero'] = pixel_spacing_zero
    metadata['pixel_spacing_low'] = pixel_spacing_low
    metadata['pixel_spacing_full'] = pixel_spacing_full

    if args.verbose:
        print('image sizes: ', ims_zero.shape, ims_low.shape, ims_full.shape)

    # FIXME: assert that number of slices are the same
    ns, nx, ny = ims_zero.shape

    idx_start = int(ns * args.discard_start_percent) # inclusive
    idx_end = int(ns * (1 - args.discard_end_percent)) # not inclusive
    idx = np.arange(idx_start, idx_end)

    metadata['slice_idx'] = idx

    if args.verbose and idx_start > 0:
        if args.verbose:
            print('discarding first {:d} slices'.format(idx_start))

    if args.verbose and idx_end < ns - 1:
        print('discarding last {:d} slices'.format(ns - idx_end))

    ims = np.stack(
    (ims_zero[idx,:,:], ims_low[idx,:,:], ims_full[idx,:,:]), axis=1)

    return ims, (hdr_zero, hdr_low, hdr_full), metadata

def _mask_images(args, ims, metadata):
    if args.verbose:
        print('masking')

    ### MASKING ###
    if not args.skip_mask:
        mask = sup.mask_im(ims, threshold=args.mask_threshold)
        metadata['mask'] = 1
        metadata['mask_threshold'] = args.mask_threshold
        ims *= mask
    else:
        metadata['mask'] = 0

    return ims, mask, metadata

def _dicom_scaling(args, ims, hdr, metadata):
    hdr_zero, hdr_low, hdr_full = hdr

    if args.scale_dicom_tags:
        if args.verbose:
            print('using dicom tags for scaling')
        rs0 = float(hdr_zero.RescaleSlope)
        ri0 = float(hdr_zero.RescaleIntercept)
        ss0 = hdr_zero[0x2005, 0x100e].value

        rs1 = float(hdr_low.RescaleSlope)
        ri1 = float(hdr_low.RescaleIntercept)
        ss1 = hdr_low[0x2005, 0x100e].value

        rs2 = float(hdr_full.RescaleSlope)
        ri2 = float(hdr_full.RescaleIntercept)
        ss2 = hdr_full[0x2005, 0x100e].value

        metadata['dicom_scaling_zero'] = (rs0, ri0, ss0)
        metadata['dicom_scaling_low'] = (rs1, ri1, ss1)
        metadata['dicom_scaling_full'] = (rs2, ri2, ss2)

        if args.verbose:
            print(rs0, rs1, rs2)
            print(ri0, ri1, ri2)
            print(ss0, ss1, ss2)

        ims[:,0,:,:] = sup.scale_slope_intercept(ims[:,0,:,:], rs0, ri0, ss0)
        ims[:,1,:,:] = sup.scale_slope_intercept(ims[:,1,:,:], rs1, ri1, ss1)
        ims[:,2,:,:] = sup.scale_slope_intercept(ims[:,2,:,:], rs2, ri2, ss2)

    return ims, metadata

def _hist_norm(args, ims, metadata):
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

    return ims, metadata

def _register(args, ims, metadata):
    spars = sitk.GetDefaultParameterMap(args.transform_type)

    if not args.skip_registration:
        metadata['reg'] = 1
        metadata['transform_type'] = args.transform_type
        ims[:,1,:,:], spars1_reg = sup.register_im(ims[:,0,:,:], ims[:,1,:,:], param_map=spars, verbose=args.verbose, im_fixed_spacing=metadata['pixel_spacing_zero'], im_moving_spacing=metadata['pixel_spacing_low'])

        if args.verbose:
            print('low dose transform parameters: {}'.format(spars1_reg[0]['TransformParameters']))

    if not args.skip_registration:
        ims[:,2,:,:], spars2_reg = sup.register_im(ims[:,0,:,:], ims[:,2,:,:], param_map=spars, verbose=args.verbose, im_fixed_spacing=metadata['pixel_spacing_zero'], im_moving_spacing=metadata['pixel_spacing_full'])

        if args.verbose:
            print('full dose transform parameters: {}'.format(spars2_reg[0]['TransformParameters']))
    else:
        metadata['reg'] = 0

    return ims, metadata

def _zoom(args, ims, metadata):
    if args.zoom:
        ims_shape = ims.shape
        if args.verbose:
            print('zooming to {}'.format(args.zoom))
        if args.verbose:
            ticz = time.time()
            print('zoom 0')
        ims_zoom_0 = zoom(ims[:,0,:,:].squeeze(), zoom=(1., args.zoom/ims_shape[2], args.zoom/ims.shape[3]), order=args.zoom_order)
        if args.verbose:
            tocz = time.time()
            print('zoom 0 done: {} s'.format(tocz-ticz))
            ticz = time.time()
            print('zoom 1')
        ims_zoom_1 = zoom(ims[:,1,:,:].squeeze(), zoom=(1., args.zoom/ims_shape[2], args.zoom/ims_shape[3]), order=args.zoom_order)
        if args.verbose:
            tocz = time.time()
            print('zoom 1 done: {} s'.format(tocz-ticz))
            ticz = time.time()
            print('zoom 2')
        ims_zoom_2 = zoom(ims[:,2,:,:].squeeze(), zoom=(1., args.zoom/ims_shape[2], args.zoom/ims_shape[3]), order=args.zoom_order)
        if args.verbose:
            tocz = time.time()
            print('zoom 2 done: {} s'.format(tocz-ticz))
        ims = np.concatenate((ims_zoom_0[:,None,...], ims_zoom_1[:,None,...], ims_zoom_2[:,None,...]), axis=1)
        if args.verbose:
            print(ims.shape)
        ns, nc, nx, ny = ims.shape
        metadata['zoom_dims'] = ims_shape
        metadata['zoom'] = args.zoom
        metadata['zoom_order'] = args.zoom_order

    return ims, metadata

def _prescale_process(args, ims, mask, metadata):
    ns, nc, nx, ny = ims.shape

    idx_scale = range(ns//2 - args.nslices // 2, ns//2 + args.nslices // 2)

    im0, im1, im2 = ims[idx_scale, 0, :, :], ims[idx_scale, 1, :, :], ims[idx_scale, 2, :, :]

    if not args.skip_mask:
        m = mask[idx_scale, 0, :, :]
    else:
        m = np.ones(im0.shape)

    im0 = im0[m != 0].ravel()
    im1 = im1[m != 0].ravel()
    im2 = im2[m != 0].ravel()

    ims_mod = np.stack((im0, im1, im2), axis=1)

    metadata['scale_slices'] = idx_scale

    return ims, ims_mod, metadata

def _match_scales(args, ims, ims_mod, metadata):
    if args.scale_matching:
        if args.verbose:
            print('intensity before scaling:')
            print('mean', np.mean(np.abs(ims_mod), axis=(0)))
            print('median', np.median(np.abs(ims_mod), axis=(0)))
            print('max', np.max(np.abs(ims_mod), axis=(0)))

        levels = np.linspace(.5, 1.5, 30)
        max_iter = 3


        ntic = time.time()
        scale_low = sup.scale_im_enhao(ims_mod[:, 0], ims_mod[:, 1], levels=levels, max_iter=max_iter)
        scale_full = sup.scale_im_enhao(ims_mod[:, 0], ims_mod[:, 1], levels=levels, max_iter=max_iter)

        metadata['scale_low'] = scale_low
        metadata['scale_full'] = scale_full

        ntoc = time.time()

        if args.verbose:
            print('scale low:', scale_low)
            print('scale full:', scale_full)
            print('done scaling data ({:.2f} s)'.format(ntoc - ntic))

        ims[:,1,:,:] = ims[:,1,:,:] * scale_low
        ims[:,2,:,:] = ims[:,2,:,:] * scale_full

        ims_mod[:,1] = ims_mod[:,1] * scale_low
        ims_mod[:,2] = ims_mod[:,2] * scale_full

        if args.verbose:
            print('intensity after scaling:')
            print('mean', np.mean(np.abs(ims_mod), axis=(0)))
            print('median', np.median(np.abs(ims_mod), axis=(0)))
            print('max', np.max(np.abs(ims_mod), axis=(0)))

    return ims, ims_mod, metadata

def _global_norm(args, ims, ims_mod, metadata):
    if args.normalize:
        normalize_fun = metadata['normalize_fun']
        if args.verbose:
            print('normalizing with function ', args.normalize_fun, normalize_fun)

        if args.joint_normalize:
            axis=(0,1)
        else:
            axis=(0)

        ntic = time.time()

        if args.global_scale_ref_im0:
            ims_norm = ims_mod[...,0]
            axis = (0)
            metadata['global_scale_ref_im0'] = True
        else:
            ims_norm = ims_mod
            metadata['global_scale_ref_im0'] = False
        scale_global = sup.normalize_scale(ims_norm, axis=axis, fun=normalize_fun)
        metadata['scale_global'] = scale_global

        if args.verbose:
            print('intensity before global scaling:')
            print('mean', np.mean(np.abs(ims_mod), axis=axis))
            print('median', np.median(np.abs(ims_mod), axis=axis))
            print('max', np.max(np.abs(ims_mod), axis=axis))

        if args.verbose:
            ntoc = time.time()
            print('global scaling:', scale_global)
            print('done ({:.2f}s)'.format(ntoc - ntic))

        if args.global_scale_ref_im0:
            ims = ims / scale_global[:,None,None,None]
        else:
            ims = ims / scale_global[:,:,None,None]
        ims_mod = ims_mod / scale_global

        if args.verbose:
            print('intensity after global scaling:')
            print('mean', np.mean(np.abs(ims_mod), axis=axis))
            print('median', np.median(np.abs(ims_mod), axis=axis))
            print('max', np.max(np.abs(ims_mod), axis=axis))

    return ims, metadata

def preprocess_chain(args):
    metadata = {}

    ims, hdr, metadata = _get_images(args, metadata)

    ims, mask, metadata = _mask_images(args, ims, metadata)
    ims, metadata = _dicom_scaling(args, ims, hdr, metadata)
    ims, metadata = _hist_norm(args, ims, metadata)
    ims, metadata = _register(args, ims, metadata)
    ims, metadata = _zoom(args, ims, metadata)

    ims, ims_mod, metadata = _prescale_process(args, ims, mask, metadata)
    ims, ims_mod, metadata = _match_scales(args, ims, ims_mod, metadata)
    ims, metadata = _global_norm(args, ims, ims_mod, metadata)

    return ims, metadata

if __name__ == '__main__':
    args = fetch_args()
    ims, metadata = preprocess_chain(args)
    suio.save_data_h5(args.out_file, data=ims, h5_key='data', metadata=metadata)
