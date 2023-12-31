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
import tempfile

import datetime
import time
import pydicom
import warnings
from glob import glob

from scipy.ndimage import zoom, gaussian_filter
from skimage.morphology import binary_closing

# try:
#     from deepbrain import Extractor as BrainExtractor
# except:
#     warnings.warn('Module deepbrain not found - cannot perform brain extraction')

import SimpleITK as sitk

import subtle.subtle_preprocess as sup
import subtle.utils.io as utils_io
import subtle.subtle_args as sargs
from subtle.utils.misc import processify

from glob import glob

def fetch_args():
    usage_str = 'usage: %(prog)s [options]'
    description_str = 'pre-process data for SubtleGad project'

    parser = sargs.parser(usage_str, description_str)
    parser.add_argument('--output', action='store', dest='out_file', type=str,
                        help='output to npy file', default='out.npy')

    args = parser.parse_args()
    return args

def assert_and_get_init_vars(args):
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

def get_images(args, metadata):
    normalize_fun, use_indiv_path, use_base_path = assert_and_get_init_vars(args)

    metadata['normalize_fun'] = normalize_fun
    metadata['use_indiv_path'] = use_indiv_path
    metadata['use_base_path'] = use_base_path

    if use_base_path:
        if args.blur_for_cs_streaks and not os.path.isdir(args.path_base):
            base_path = args.path_base.replace('_blur', '')
        else:
            base_path = args.path_base

        dicom_dirs = utils_io.get_dicom_dirs(base_path, override=args.override)

        ### only for BCH
        # metadata['inference_only'] = True
        # dicom_dirs = sorted([d for d in glob('{}/*'.format(base_path)) if os.path.isdir(d)])
        # args.path_zero = dicom_dirs[0]
        # args.path_low = dicom_dirs[1]
        # args.path_full = args.path_low
        # if len(dicom_dirs) == 3:
        #     args.path_zero = [d for d in dicom_dirs if 'mprage' in d.lower()][0]
        #     args.path_low = [
        #         d for d in dicom_dirs if 'space' in d.lower() and 'reg' in d.lower()
        #     ][0]
        #     args.path_full = args.path_low
        # else:
        #     args.path_zero = [
        #         d for d in dicom_dirs
        #         if 'mprage' in d.lower() and 'smr' in d.lower()
        #     ][0]
        #     args.path_low = [
        #         d for d in dicom_dirs if 'space' in d.lower() and 'reg' in d.lower()
        #     ][0]
        #     args.path_full = args.path_low
        ### only for BCH

        args.path_zero = dicom_dirs[0]
        args.path_low = dicom_dirs[1]

        if len(dicom_dirs) == 3:
            args.path_full = dicom_dirs[2]
            metadata['inference_only'] = False
        else:
            args.path_full = args.path_low
            metadata['inference_only'] = True

        if args.verbose:
            print('path_zero = {}'.format(args.path_zero))
            print('path_low = {}'.format(args.path_low))
            print('path_full = {}'.format(args.path_full))

    ims_zero, hdr_zero = utils_io.dicom_files(args.path_zero, normalize=False)
    ims_low, hdr_low = utils_io.dicom_files(args.path_low, normalize=False)
    ims_full, hdr_full = utils_io.dicom_files(args.path_full, normalize=False)

    pixel_spacing_zero = utils_io.get_pixel_spacing(hdr_zero)
    pixel_spacing_low = utils_io.get_pixel_spacing(hdr_low)
    pixel_spacing_full = utils_io.get_pixel_spacing(hdr_full)

    metadata['pixel_spacing_zero'] = pixel_spacing_zero
    metadata['pixel_spacing_low'] = pixel_spacing_low
    metadata['pixel_spacing_full'] = pixel_spacing_full

    if args.verbose:
        print('image sizes: ', ims_zero.shape, ims_low.shape, ims_full.shape)

    nslices = [ims_zero.shape[0], ims_low.shape[0], ims_full.shape[0]]

    if ims_zero.shape[1] != ims_low.shape[1]:
        ims_low = sup.center_crop(ims_low, ims_zero)
        ims_full = sup.center_crop(ims_full, ims_zero)
    elif len(set(nslices)) > 1:
        n_pad = np.max(nslices)
        ims_zero = np.pad(ims_zero, pad_width=[(n_pad - nslices[0], 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
        ims_low = np.pad(ims_low, pad_width=[(n_pad - nslices[1], 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
        ims_full = np.pad(ims_full, pad_width=[(n_pad - nslices[2], 0), (0, 0), (0, 0)], mode='constant', constant_values=0)

    if args.verbose:
        print('image sizes after resize: ', ims_zero.shape, ims_low.shape, ims_full.shape)

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

def mask_images(args, ims, metadata):
    if args.verbose:
        print('masking')

    ### MASKING ###
    if not args.skip_mask:
        mask = sup.mask_im(ims, threshold=args.mask_threshold, noise_mask_area=args.noise_mask_area)
        metadata['mask'] = 1
        metadata['mask_threshold'] = args.mask_threshold
        ims *= mask
        metadata['lambda'].append({
            'name': 'mask_images',
            'fn': lambda images: images * sup.mask_im(images, threshold=args.mask_threshold, noise_mask_area=args.noise_mask_area)
        })
    else:
        mask = np.ones_like(ims)
        metadata['mask'] = 0

    return ims, mask, metadata

def dicom_scaling(args, ims, hdr, metadata):
    hdr_zero, hdr_low, hdr_full = hdr

    if args.scale_dicom_tags and 'RescaleSlope' in hdr_zero:
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

        rs = [rs0, rs1, rs2]
        ri = [ri0, ri1, ri2]
        ss = [ss0, ss1, ss2]

        ssi = lambda idx: (lambda images: sup.scale_slope_intercept(images[:,idx,:,:], rs[idx], ri[idx], ss[idx]))

        for cont in np.arange(3):
            ims[:, cont, :, :] = ssi(cont)(ims)

        metadata['lambda'].append({
            'name': 'dicom_scaling',
            'fn': [ssi(idx) for idx in np.arange(3)]
        })

    return ims, metadata

def hist_norm(args, ims, metadata):
    if not args.skip_hist_norm:
        if args.verbose:
            print('Histogram normalization')
        metadata['hist_norm'] = 1
        # FIXME: expose to outside world. subject to change once we implement white striping
        levels=1024
        points=50
        mean_intensity=True

        print('histogram data type', ims.dtype)

        hnorm = lambda idx: (lambda images: sup.scale_im(images[:, 0, :, :], images[:, idx, :, :], levels, points, mean_intensity))

        eye = lambda images: images[:, 0, :, :]

        ims[:,1,:,:] = hnorm(1)(ims)
        ims[:,2,:,:] = hnorm(2)(ims)

        metadata['lambda'].append({
            'name': 'hist_norm',
            'fn': [eye, hnorm(1), hnorm(2)]
        })
    else:
        if args.verbose:
            print('Skipping histogram normalization')
        metadata['hist_norm'] = 0

    return ims, metadata

def register(args, ims, metadata):
    spars = sitk.GetDefaultParameterMap(args.transform_type, args.reg_n_levels)

    if not args.skip_registration:
        metadata['reg'] = 1
        metadata['transform_type'] = args.transform_type

        zero_stk = None
        low_stk = None
        full_stk = None

        spacing_zero = metadata['pixel_spacing_zero']
        spacing_low = metadata['pixel_spacing_low']
        spacing_full = metadata['pixel_spacing_full']

        if args.register_with_dcm_reference:
            zero_stk = sup.dcm_to_sitk(args.path_zero)
            low_stk = sup.dcm_to_sitk(args.path_low)
            full_stk = sup.dcm_to_sitk(args.path_full)

            spacing_zero = None
            spacing_low = None
            spacing_full = None

            print('registering with dcm reference')
            print(args.path_zero)
            print(args.path_low)
            print(args.path_full)

            low_shape = list(low_stk.GetSize()[::-1])
            full_shape = list(full_stk.GetSize()[::-1])

        if args.external_reg_ref:
            print('Using external registration reference...')
            cnum = args.path_zero.split('/')[-2]
            ext_bpath = os.path.join(args.external_reg_ref, cnum)
            ext_reg_ref = [s for s in glob('{}/*'.format(ext_bpath))][0]
            old_zero_stk = zero_stk
            zero_stk = sup.dcm_to_sitk(ext_reg_ref)

            ref_npy = sitk.GetArrayFromImage(zero_stk)
            ref_npy = ref_npy / ref_npy.mean()

            reg_pre, _ = sup.register_im(ref_npy, ims[:, 0], param_map=spars, verbose=args.verbose, im_fixed_spacing=spacing_zero, im_moving_spacing=spacing_low, non_rigid=args.non_rigid_reg, ref_fixed=zero_stk, ref_moving=old_zero_stk)

            reg_low, _ = sup.register_im(ref_npy, ims[:, 1], param_map=spars, verbose=args.verbose, im_fixed_spacing=spacing_zero, im_moving_spacing=spacing_low, non_rigid=args.non_rigid_reg, ref_fixed=zero_stk, ref_moving=low_stk)

            reg_full, _ = sup.register_im(ref_npy, ims[:, 2], param_map=spars, verbose=args.verbose, im_fixed_spacing=spacing_zero, im_moving_spacing=spacing_full, non_rigid=args.non_rigid_reg,
            ref_fixed=zero_stk, ref_moving=full_stk)

            ims = np.array([reg_pre, reg_low, reg_full]).transpose(1, 0, 2, 3)
        else:
            ref_npy = ims[:, 0]

            stk_ref_imgs = [zero_stk, low_stk, full_stk]

            ims[:, 1], spars1_reg = sup.register_im(ref_npy, ims[:, 1], param_map=spars, verbose=args.verbose, im_fixed_spacing=spacing_zero, im_moving_spacing=spacing_low, non_rigid=args.non_rigid_reg,
            ref_fixed=zero_stk, ref_moving=low_stk)

            if args.verbose:
                print('low dose transform parameters: {}'.format(spars1_reg[0]['TransformParameters']))

            ims[:, 2], spars2_reg = sup.register_im(ref_npy, ims[:, 2], param_map=spars, verbose=args.verbose, im_fixed_spacing=spacing_zero, im_moving_spacing=spacing_full, non_rigid=args.non_rigid_reg,
            ref_fixed=zero_stk, ref_moving=full_stk)

            reg_params = [None, spars1_reg, spars2_reg]
            spacing_keys = ['pixel_spacing_zero', 'pixel_spacing_low', 'pixel_spacing_full']

            if args.use_fsl_reg:
                print('Planning to apply registration params computed from skull stripped images...')
                reg_transform = lambda idx: (
                    lambda images: sup.apply_reg_transform(
                        images[:, idx, :, :], metadata[spacing_keys[idx]], reg_params[idx],
                        ref_img=stk_ref_imgs[0]
                    )
                )
            else:
                print('Planning to re-run registration on full brain images...')
                reg_transform = lambda idx: (
                    lambda images: sup.register_im(
                        images[:, 0, :, :,], images[:, idx, :, :],
                        param_map=spars, verbose=args.verbose,
                        im_fixed_spacing=metadata['pixel_spacing_zero'],
                        im_moving_spacing=metadata[spacing_keys[idx]],
                        non_rigid=args.non_rigid_reg,
                        return_params=False, ref_fixed=zero_stk, ref_moving=stk_ref_imgs[idx]
                    )
                )

            eye = lambda images: images[:, 0, :, :]

            metadata['lambda'].append({
                'name': 'register',
                'fn': [eye, reg_transform(1), reg_transform(2)]
            })

            if args.verbose:
                print('full dose transform parameters: {}'.format(spars2_reg[0]['TransformParameters']))
    else:
        metadata['reg'] = 0

    return ims, metadata

def breast_processing(args, ims, metadata):
    if args.breast_gad:
        print('HERE inside breast processing...')
        cnum = args.path_zero.split('/')[-2]
        mask = np.load('{}/{}.npy'.format(args.breast_mask_path, cnum))

        ims[:, 0] = ims[:, 0] * mask
        ims[:, 1] = ims[:, 1] * mask
        ims[:, 2] = ims[:, 2] * mask

    return ims, metadata


def zoom_process(args, ims, metadata):
    if args.zoom:
        ims_shape = ims.shape
        print('Zoom processing...')

        ims_zoom_0 = zoom(ims[:,0,:,:].squeeze(), zoom=(1., args.zoom/ims_shape[2], args.zoom/ims.shape[3]), order=args.zoom_order)

        ims_zoom_1 = zoom(ims[:,1,:,:].squeeze(), zoom=(1., args.zoom/ims_shape[2], args.zoom/ims_shape[3]), order=args.zoom_order)

        ims_zoom_2 = zoom(ims[:,2,:,:].squeeze(), zoom=(1., args.zoom/ims_shape[2], args.zoom/ims_shape[3]), order=args.zoom_order)

        imzoom = lambda idx: (lambda images: zoom(images[:,idx,:,:].squeeze(), zoom=(1., args.zoom/ims_shape[2], args.zoom/images.shape[3]), order=args.zoom_order))

        ims = np.concatenate((imzoom(0)(ims), imzoom(1)(ims), imzoom(2)(ims)), axis=1)

        metadata['lambda'].append({
            'name': 'zoom_process',
            'fn': [imzoom(idx) for idx in np.arange(3)]
        })

        if args.verbose:
            print(ims.shape)

        ns, nc, nx, ny = ims.shape
        metadata['zoom_dims'] = ims_shape
        metadata['zoom'] = args.zoom
        metadata['zoom_order'] = args.zoom_order

    return ims, metadata

def prescale_process(args, ims, mask, metadata):
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

def match_scales(args, ims, ims_mod, metadata):
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
        scale_full = sup.scale_im_enhao(ims_mod[:, 0], ims_mod[:, 2], levels=levels, max_iter=max_iter)

        metadata['scale_low'] = scale_low
        metadata['scale_full'] = scale_full

        ntoc = time.time()

        if args.verbose:
            print('scale low:', scale_low)
            print('scale full:', scale_full)
            print('done scaling data ({:.2f} s)'.format(ntoc - ntic))

        scales = [1, scale_low, scale_full]

        imscale = lambda idx: (lambda images: images[:, idx, :, :] * scales[idx])

        ims[:,0,:,:] = imscale(0)(ims)
        ims[:,1,:,:] = imscale(1)(ims)
        ims[:,2,:,:] = imscale(2)(ims)

        ims_mod[:,1] = ims_mod[:,1] * scale_low
        ims_mod[:,2] = ims_mod[:,2] * scale_full

        metadata['lambda'].append({
            'name': 'match_scales',
            'fn': [imscale(idx) for idx in np.arange(3)]
        })

        if args.verbose:
            print('intensity after scaling:')
            print('mean', np.mean(np.abs(ims_mod), axis=(0)))
            print('median', np.median(np.abs(ims_mod), axis=(0)))
            print('max', np.max(np.abs(ims_mod), axis=(0)))

    return ims, ims_mod, metadata

def global_norm(args, ims, ims_mod, metadata):
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
            metadata['lambda'].append({
                'name': 'global_norm',
                'fn': lambda images: images / scale_global[:, None, None, None]
            })
        else:
            ims = ims / scale_global[:,:,None,None]
            metadata['lambda'].append({
                'name': 'global_norm',
                'fn': lambda images: images / scale_global[:, :, None, None]
            })
        ims_mod = ims_mod / scale_global

        if args.verbose:
            print('intensity after global scaling:')
            print('mean', np.mean(np.abs(ims_mod), axis=axis))
            print('median', np.median(np.abs(ims_mod), axis=axis))
            print('max', np.max(np.abs(ims_mod), axis=axis))

    return ims, metadata

def _mask_npy(img_npy):
    ext = BrainExtractor()

    img_scale = np.interp(img_npy, (img_npy.min(), img_npy.max()), (0, 1))
    segment_probs = ext.run(img_scale)

    # TODO - binary fill holes
    mask = sup.get_largest_connected_component(segment_probs > 0.5)
    return mask

def fsl_brain_mask(args, ims):
    return _brain_mask(args, ims)

@processify
def fsl_brain_mask_processify(args, ims):
    return _brain_mask(args, ims)

def _brain_mask(args, ims):
    mask = None

    if args.fsl_mask:
        print('Extracting brain regions using deepbrain...')

        with tempfile.TemporaryDirectory() as tmp:
            if args.verbose:
                print('BET Zero')
            ## Temporarily using DL based method for extraction
            mask_zero = _mask_npy(ims[:, 0, ...])

            if args.fsl_mask_all_ims:
                if args.verbose:
                    print('BET Low')
                mask_low = _mask_npy(ims[:, 1, ...])

                if args.verbose:
                    print('BET Full')
                mask_full = _mask_npy(ims[:, 2, ...])

                if args.union_brain_masks:
                # union of all masks
                    mask = ((mask_zero > 0 ) | (mask_low > 0) | (mask_full > 0))
                else:
                    mask = np.array([mask_zero, mask_low, mask_full]).transpose(1, 0, 2, 3)
            else:
                mask = mask_zero
    return mask

def apply_fsl_mask(args, ims, fsl_mask):
    ims_mask = np.copy(ims)
    if args.fsl_mask:
        print('Applying computed FSL masks on images. Mask shape -', fsl_mask.shape)
        ims_mask = np.zeros_like(ims)

        if fsl_mask.ndim == 4:
            ims_mask = fsl_mask * ims
        else:
            for cont in range(ims.shape[1]):
                ims_mask[:, cont, :, :] = fsl_mask * ims[:, cont, :, :]

    return ims_mask

def fsl_reject_slices(args, ims, fsl_mask, metadata):
    if args.fsl_area_threshold_cm2 is not None:
        print('Removing slices where brain area is less than {}cm2'.format(args.fsl_area_threshold_cm2))

        dicom_spacing = metadata['new_spacing'] if 'new_spacing' in metadata else metadata['pixel_spacing_zero']

        fsl_mask_copy = np.copy(fsl_mask)

        if fsl_mask_copy.ndim == 4:
            fsl_mask_copy = fsl_mask_copy[:, 0]

        mask_areas = np.array([sup.get_brain_area_cm2(mask_slice, dicom_spacing) for mask_slice in fsl_mask])

        good_slice_idx = (mask_areas >= args.fsl_area_threshold_cm2)

        metadata['good_slice_indices'] = good_slice_idx
        print('{} slices retained'.format(good_slice_idx.sum()))

    return ims, metadata

def _get_spacing_from_dicom(dirpath_dicom):
    fpath_dicom = [
        fpath for fpath in glob('{}/**/*'.format(dirpath_dicom), recursive=True)
        if os.path.isfile(fpath)
    ][0]

    dicom = pydicom.dcmread(fpath_dicom)
    return np.array([
        float(dicom.SliceThickness),
        float(dicom.PixelSpacing[0]),
        float(dicom.PixelSpacing[1])
    ])

def resample_isotropic(args, ims, metadata):
    metadata['original_size'] = (ims.shape[2], ims.shape[3])

    if args.resample_isotropic > 0:
        print('Resampling images to {}mm isotropic...'.format(args.resample_isotropic))
        print('Current image shapes...', ims[:, 0, ...].shape)
        new_spacing = [args.resample_isotropic] * 3
        # new_spacing = [
        #     metadata['pixel_spacing_zero'][-1], args.resample_isotropic, args.resample_isotropic
        # ]

        spacing_zero = _get_spacing_from_dicom(args.path_zero)
        spacing_low = _get_spacing_from_dicom(args.path_low)
        spacing_full = _get_spacing_from_dicom(args.path_full)

        metadata['old_spacing_zero'] = spacing_zero
        metadata['old_spacing_low'] = spacing_low
        metadata['old_spacing_full'] = spacing_full

        ims_zero, ims_low, ims_full = np.transpose(np.copy(ims), (1, 0, 2, 3))

        # metadata['new_spacing'] = [0.5, spacing_zero[1], spacing_zero[2]]
        #
        # print('Resampling zero dose...')
        # ims_zero, new_spacing = sup.zoom_iso(ims_zero, spacing_zero, [0.5, spacing_zero[1], spacing_zero[2]])
        # metadata['new_spacing'] = new_spacing
        #
        # print('Resampling low dose...')
        # ims_low, _ = sup.zoom_iso(ims_low, spacing_low, [0.5, spacing_low[1], spacing_low[2]])
        #
        # print('Resampling full dose...')
        # ims_full, _ = sup.zoom_iso(ims_full, spacing_full, [0.5, spacing_full[1], spacing_full[2]])

        metadata['new_spacing'] = new_spacing

        print('Resampling zero dose...')
        ims_zero, new_spacing = sup.zoom_iso(ims_zero, spacing_zero, new_spacing)
        metadata['new_spacing'] = new_spacing

        print('Resampling low dose...')
        ims_low, _ = sup.zoom_iso(ims_low, spacing_low, new_spacing)

        if metadata['inference_only']:
            ims_full = ims_low
        else:
            print('Resampling full dose...')
            ims_full, _ = sup.zoom_iso(ims_full, spacing_full, new_spacing)

        print('New image shape', ims_zero.shape)
        metadata['resampled_size'] = (ims_zero.shape[1], ims_zero.shape[2])

        ims_iso = np.transpose(
            np.array([ims_zero, ims_low, ims_full]),
            (1, 0, 2, 3)
        )
        return ims_iso, metadata
    else:
        metadata['resampled_size'] = metadata['original_size']
        metadata['new_spacing'] = metadata['pixel_spacing_zero']
        metadata['old_spacing_low'] = metadata['pixel_spacing_low']
    return ims, metadata

def reshape_fsl_mask(args, fsl_mask, metadata):
    fsl_reshape = np.copy(fsl_mask)

    if args.fsl_mask and args.resample_isotropic > 0:
        print('reshaping fsl mask')
        fsl_mask_ims = np.zeros((fsl_mask.shape[0], 3, fsl_mask.shape[1], fsl_mask.shape[2]))

        if fsl_mask.ndim == 4:
            fsl_mask_ims = np.copy(fsl_mask)
        else:
            fsl_mask_ims[:, 0, ...] = np.copy(fsl_mask)
            fsl_mask_ims[:, 1, ...] = np.copy(fsl_mask)
            fsl_mask_ims[:, 2, ...] = np.copy(fsl_mask)

        fsl_reshape, _ = resample_isotropic(args, fsl_mask_ims, metadata)
        if fsl_mask.ndim == 3:
            fsl_reshape = fsl_reshape[:, 0, ...]
        fsl_reshape = (fsl_reshape >= 0.5).astype(fsl_mask.dtype)
    return fsl_reshape

def apply_preprocess(args, ims, unmasked_ims, metadata):
    if not args.fsl_mask:
        del metadata['lambda']
        return ims, metadata

    print('Applying all preprocessing steps on unmasked images...')
    for step in metadata['lambda']:
        print('Applying {} on unmasked images...'.format(step['name']))
        if callable(step['fn']):
            unmasked_ims = step['fn'](unmasked_ims)
        elif isinstance(step['fn'], list):
            for idx, fn in enumerate(step['fn']):
                unmasked_ims[:, idx, :, :] = fn(unmasked_ims)
        else:
            continue

    del metadata['lambda']
    return unmasked_ims, metadata

def zero_pad(args, ims, metadata):
    if args.pad_for_size > 0:
        ims_pad = sup.zero_pad(ims, target_size=args.pad_for_size)
        print('Shape after zero padding...', ims_pad.shape)
        metadata['zero_pad_size'] = (ims_pad.shape[2], ims_pad.shape[3])

        return ims_pad, metadata

    return ims, metadata

def calc_img_statistics(unmasked_ims, ims, metadata):
    print(unmasked_ims.shape, ims.shape)
    percentiles = [5, 50, 95, 99, 100]
    p1 = np.percentile(unmasked_ims, percentiles, axis=(0,2,3))
    p2 = np.percentile(ims, percentiles, axis=(0,2,3))
    metadata['percentiles'] = percentiles
    metadata['unmasked_ims_percentiles'] = p1
    metadata['ims_percentiles'] = p2
    return metadata

def gaussian_blur_input(args, ims, metadata):
    if not args.blur_for_cs_streaks:
        return ims, metadata

    blur_fn = lambda x: gaussian_filter(x, sigma=[0, 1.5])
    ims_low = ims[:, 1, ...]
    tr_dict = {
        'AX': lambda x: x,
        'SAG': lambda x: x.transpose(1, 0, 2),
        'COR': lambda x: x.transpose(2, 0, 1)
    }

    ims_low = tr_dict[args.acq_plane](ims_low)
    ims_blur = np.array([blur_fn(sl) for sl in ims_low])
    ims_blur = tr_dict[args.acq_plane](ims_blur)
    ims[:, 1, ...] = ims_blur

    return ims, metadata


def preprocess_chain(args, metadata={'lambda': []}):
    ims, hdr, metadata = get_images(args, metadata)
    unmasked_ims = np.copy(ims)

    # first apply a mask to remove regions with zero signal
    ims, mask, metadata = mask_images(args, ims, metadata)

    # next apply a BET mask to remove non-brain tissue
    if not args.dicom_inference:
        fsl_mask = fsl_brain_mask(args, ims)
    else:
        fsl_mask = fsl_brain_mask_processify(args, ims)
    ims = apply_fsl_mask(args, ims, fsl_mask)

    # scale and register images based on BET images
    ims, metadata = dicom_scaling(args, ims, hdr, metadata)
    ims, metadata = register(args, ims, metadata)
    ims, metadata = breast_processing(args, ims, metadata)
    ims, metadata = hist_norm(args, ims, metadata)
    ims, metadata = zoom_process(args, ims, metadata)

    ims, ims_mod, metadata = prescale_process(args, ims, mask, metadata)
    ims, ims_mod, metadata = match_scales(args, ims, ims_mod, metadata)
    ims, metadata = global_norm(args, ims, ims_mod, metadata)

    # reapply the preprocessing chain to the unmasked images
    unmasked_ims, metadata = apply_preprocess(args, ims, unmasked_ims, metadata)

    ims, _ = gaussian_blur_input(args, ims, metadata)
    unmasked_ims, metadata = gaussian_blur_input(args, unmasked_ims, metadata)


    ims, _ = resample_isotropic(args, ims, metadata) # dont save to metadata on masked ims
    if args.fsl_mask:
        unmasked_ims, metadata = resample_isotropic(args, unmasked_ims, metadata)
    else:
        unmasked_ims = ims
    fsl_mask = reshape_fsl_mask(args, fsl_mask, metadata)

    ims, metadata = fsl_reject_slices(args, ims, fsl_mask, metadata)

    ims, _ = zero_pad(args, ims, metadata) # dont save to metadata on masked ims
    unmasked_ims, metadata = zero_pad(args, unmasked_ims, metadata)

    # calculate and save some useful image statistics
    metadata = calc_img_statistics(unmasked_ims, ims, metadata)

    return unmasked_ims, ims, metadata

def execute_chain(args):
    print('------')
    print(args.debug_print())
    print('------\n\n\n')
    ims, ims_mask, metadata = preprocess_chain(args)
    save_data(args, ims, ims_mask, metadata)

def save_data(args, ims, ims_mask, metadata=None):
    if args.file_ext == 'h5':
        utils_io.save_data_h5(args.out_file, data=ims, data_mask=ims_mask, h5_key='data')
    else:
        npy_data = np.array([ims, ims_mask])
        utils_io.save_data_npy(args.out_file, npy_data)
    if metadata is not None:
        utils_io.save_meta_h5(args.out_file.replace('.{}'.format(args.file_ext), '_meta.h5'), metadata)

def preprocess_multi_contrast(args):
    ### Init and fetch data
    case_num = args.path_base.split('/')[-1]
    fpath_t1 = os.path.join(args.data_dir, '{}.{}'.format(case_num, args.file_ext))
    if not os.path.exists(fpath_t1):
        rep = 'h5' if args.file_ext == 'npy' else 'npy'
        fpath_t1 = fpath_t1.replace(args.file_ext, rep)
    t1_data = utils_io.load_file(fpath_t1, params={'h5_key': 'all'}).astype(np.float32)

    mc_kw = args.multi_contrast_kw.split(',')
    dcmdir_mc = utils_io.get_dcmdir_with_kw(args.path_base, mc_kw)
    assert dcmdir_mc is not None, 'Study does not have a valid scan with keywords {}'.format(mc_kw)

    t1_dirs = utils_io.get_dicom_dirs(args.path_base, override=args.override)
    if len(t1_dirs) == 3:
        dcmdir_t1_pre, dcmdir_t1_low, dcmdir_t1_full = t1_dirs
    else:
        dcmdir_t1_pre, dcmdir_t1_full = t1_dirs
        dcmdir_t1_low = dcmdir_t1_full
    mc_vol, mc_hdr = utils_io.dicom_files(dcmdir_mc)
    t1_pre = t1_data[0, :, 0]
    t1_low = t1_data[0, :, 1]

    ### Noise masking
    print('Noise masking...')
    noise_mask = sup.mask_im(np.array([mc_vol]), threshold=args.mask_threshold, noise_mask_area=args.noise_mask_area)[0]
    mc_vol = mc_vol * noise_mask

    ### Registration
    print('Registration...')
    t1_spacing = _get_spacing_from_dicom(dcmdir_t1_pre)
    mc_spacing = _get_spacing_from_dicom(dcmdir_mc)

    reg_pmap = sitk.GetDefaultParameterMap(args.transform_type)
    ref_fixed = sup.dcm_to_sitk(dcmdir_t1_pre)
    ref_moving = sup.dcm_to_sitk(dcmdir_mc)

    ref_z, ref_x, ref_y = ref_fixed.GetSize()[::-1]
    t1_pre_ref = t1_pre.copy()

    if ref_z > t1_pre.shape[0]:
        diff_z = (ref_z - t1_pre.shape[0]) // 2
        t1_pre_ref = np.pad(t1_pre, pad_width=[(diff_z, diff_z), (0, 0), (0, 0)],
                        mode='constant', constant_values=0)
    elif ref_z < t1_pre.shape[0]:
        t1_pre_ref = sup.center_crop(t1_pre, np.zeros((ref_z, ref_x, ref_y)))

    print('Multi contrast volume shape before registration', mc_vol.shape)
    mc_vol, _ = sup.register_im(
        t1_pre_ref, mc_vol, param_map=reg_pmap, im_fixed_spacing=t1_spacing,
        im_moving_spacing=mc_spacing, ref_fixed=ref_fixed, ref_moving=ref_moving
    )

    print('Multi contrast volume after registration before crop/pad', mc_vol.shape)

    if mc_vol.shape[0] > t1_pre.shape[0]:
        mc_vol = sup.center_crop(mc_vol, np.zeros((t1_pre.shape[0], t1_pre.shape[1], t1_pre.shape[2])))
    elif mc_vol.shape[0] < t1_pre.shape[0]:
        diff = (t1_pre.shape[0] - mc_vol.shape[0])
        pad_z = diff // 2
        if pad_z == 0:
            padw = (1, 0)
        elif diff % 2 != 0:
            padw = (pad_z, pad_z + 1)
        else:
            padw = (pad_z, pad_z)
        mc_vol = np.pad(mc_vol, pad_width=[padw, (0, 0), (0, 0)],
                        mode='constant', constant_values=0)

    print('Multi contrast volume shape after crop/pad', mc_vol.shape)

    ### Skull stripping
    print('Skull stripping...')
    mask = t1_data[1, :, 2] >= 0.1
    mask = binary_closing(mask)

    print('mc_vol, mask', mc_vol.shape, mask.shape)
    mc_vol_mask = mc_vol * mask

    ## Scaling
    print('Scaling...')
    mc_vol = mc_vol / mc_vol.mean()
    mc_vol_mask = mc_vol_mask / mc_vol_mask.mean()
    # mc_vol = np.interp(mc_vol, (mc_vol.min(), mc_vol.max()), (t1_pre.min(), t1_pre.max()))
    # mc_vol_mask = np.interp(mc_vol_mask, (mc_vol_mask.min(), mc_vol_mask.max()), (t1_pre.min(), t1_pre.max()))

    # No histogram equalization for FLAIR
    # print('Histogram equalization...')
    # mc_vol = sup.scale_im(t1_low, mc_vol.astype(t1_low.dtype))

    # if 't2' in mc_kw:
    #     mc_vol *= args.t2_scaling_constant

    ### Saving data
    save_data(args, mc_vol, mc_vol_mask, metadata=None)


if __name__ == '__main__':
    args = fetch_args()

    if args.multi_contrast_mode:
        preprocess_multi_contrast(args)
    else:
        execute_chain(args)
