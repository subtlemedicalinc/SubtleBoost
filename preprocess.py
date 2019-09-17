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

from scipy.ndimage import zoom
from nipype.interfaces import fsl
fsl.FSLCommand.set_default_output_type('NIFTI')


sys.path.insert(0, '/home/subtle/jon/tools/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.5/SimpleITK')
import SimpleITK as sitk

import subtle.subtle_preprocess as sup
import subtle.subtle_io as suio
import subtle.subtle_args as sargs


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
        dicom_dirs = suio.get_dicom_dirs(args.path_base, override=args.override)

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
    if not args.skip_scale_im:
        metadata['hist_norm'] = 1
        # FIXME: expose to outside world. subject to change once we implement white striping
        levels=1024
        points=50
        mean_intensity=True

        hnorm = lambda idx: (lambda images: sup.scale_im(images[:, 0, :, :], images[:, idx, :, :], levels, points, mean_intensity))

        eye = lambda images: images[:, 0, :, :]

        ims[:,1,:,:] = hnorm(1)(ims)
        ims[:,2,:,:] = hnorm(2)(ims)

        metadata['lambda'].append({
            'name': 'hist_norm',
            'fn': [eye, hnorm(1), hnorm(2)]
        })
    else:
        metadata['hist_norm'] = 0

    return ims, metadata

def register(args, ims, metadata):
    spars = sitk.GetDefaultParameterMap(args.transform_type)

    if not args.skip_registration:
        metadata['reg'] = 1
        metadata['transform_type'] = args.transform_type

        ims[:, 1, :, :], spars1_reg = sup.register_im(ims[:, 0, :, :], ims[:, 1, :, :], param_map=spars, verbose=args.verbose, im_fixed_spacing=metadata['pixel_spacing_zero'], im_moving_spacing=metadata['pixel_spacing_low'])


        if args.verbose:
            print('low dose transform parameters: {}'.format(spars1_reg[0]['TransformParameters']))

        ims[:, 2, :, :], spars2_reg = sup.register_im(ims[:, 0, :, :], ims[:, 2, :, :], param_map=spars, verbose=args.verbose, im_fixed_spacing=metadata['pixel_spacing_zero'], im_moving_spacing=metadata['pixel_spacing_full'])

        reg_params = [None, spars1_reg, spars2_reg]
        spacing_keys = ['pixel_spacing_zero', 'pixel_spacing_low', 'pixel_spacing_full']

        reg_transform = lambda idx: (lambda images: sup.apply_reg_transform(images[:, idx, :, :], metadata[spacing_keys[idx]], reg_params[idx]))

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
        scale_full = sup.scale_im_enhao(ims_mod[:, 0], ims_mod[:, 1], levels=levels, max_iter=max_iter)

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

def _mask_nii(fpath_nii, outdir, fsl_threshold):
    bet_out_name = '{}_bet'.format(fpath_nii.split('/')[-1].replace('.nii', ''))
    bet_outfile = '{}/{}.nii'.format(outdir, bet_out_name)

    bet_node = fsl.BET(frac=fsl_threshold, reduce_bias=False, mask=True)

    bet_node.inputs.in_file = fpath_nii
    bet_node.inputs.out_file = bet_outfile

    res = bet_node.run()

    # FIXME: use res.outputs.mask_file ?
    mask = sup.nii2npy('{}/{}_mask.nii'.format(outdir, bet_out_name))

    return mask

def fsl_brain_mask(args):
    mask = None

    if args.fsl_mask:
        print('Extracting brain regions using FSL...')

        with tempfile.TemporaryDirectory() as tmp:
            if args.verbose:
                print('BET Zero')
            out_zero = sup.dcm2nii(args.path_zero, tmp)
            mask_zero = _mask_nii(out_zero, tmp, args.fsl_threshold)

            if args.fsl_mask_all_ims:
                if args.verbose:
                    print('BET Low')
                out_low = sup.dcm2nii(args.path_low, tmp)
                mask_low = _mask_nii(out_low, tmp, args.fsl_threshold)

                if args.verbose:
                    print('BET Full')
                out_full = sup.dcm2nii(args.path_full, tmp)
                mask_full = _mask_nii(out_full, tmp, args.fsl_threshold)

                # union of all masks
                mask = ((mask_zero > 0 ) | (mask_low > 0) | (mask_full > 0))
            else:
                mask = mask_zero
    return mask

def apply_fsl_mask(args, ims, fsl_mask):
    ims_mask = np.copy(ims)
    if args.fsl_mask:
        print('Applying computed FSL masks on images')
        ims_mask = np.zeros_like(ims)
        for cont in range(ims.shape[1]):
            ims_mask[:, cont, :, :] = fsl_mask * ims[:, cont, :, :]

    return ims_mask

def fsl_reject_slices(args, ims, fsl_mask, metadata):
    if args.fsl_area_threshold_cm2 is not None:
        print('Removing slices where brain area is less than {}cm2'.format(args.fsl_area_threshold_cm2))

        dicom_spacing = metadata['new_spacing'] if 'new_spacing' in metadata else metadata['pixel_spacing_zero']

        mask_areas = np.array([sup.get_brain_area_cm2(mask_slice, dicom_spacing) for mask_slice in fsl_mask])

        good_slice_idx = (mask_areas >= args.fsl_area_threshold_cm2)

        metadata['good_slice_indices'] = good_slice_idx
        print('{} slices retained'.format(good_slice_idx.sum()))

    return ims, metadata

def _get_spacing_from_dicom(dirpath_dicom):
    fpath_dicom = [fpath for fpath in glob('{}/**/*.dcm'.format(dirpath_dicom), recursive=True)][0]

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

        spacing_zero = _get_spacing_from_dicom(args.path_zero)
        spacing_low = _get_spacing_from_dicom(args.path_low)
        spacing_full = _get_spacing_from_dicom(args.path_full)

        metadata['old_spacing_zero'] = spacing_zero
        metadata['old_spacing_low'] = spacing_low
        metadata['old_spacing_full'] = spacing_full

        ims_zero, ims_low, ims_full = np.transpose(np.copy(ims), (1, 0, 2, 3))

        metadata['new_spacing'] = new_spacing

        print('New spacing...', new_spacing)

        print('Resampling zero dose...')
        ims_zero, new_spacing = sup.zoom_iso(ims_zero, spacing_zero, new_spacing)
        metadata['new_spacing'] = new_spacing

        print('Resampling low dose...')
        ims_low, _ = sup.zoom_iso(ims_low, spacing_low, new_spacing)

        print('Resampling full dose...')
        ims_full, _ = sup.zoom_iso(ims_full, spacing_full, new_spacing)

        print('New image shape', ims_zero.shape)
        metadata['resampled_size'] = (ims_zero.shape[1], ims_zero.shape[2])

        return np.transpose(
            np.array([ims_zero, ims_low, ims_full]),
            (1, 0, 2, 3)
        )

    return ims, metadata

def reshape_fsl_mask(args, fsl_mask, metadata):
    fsl_reshape = np.copy(fsl_mask)

    if args.fsl_mask and args.resample_isotropic > 0:
        print('reshaping fsl mask')
        fsl_mask_ims = np.zeros((fsl_mask.shape[0], 3, fsl_mask.shape[1], fsl_mask.shape[2]))

        fsl_mask_ims[:, 0, ...] = np.copy(fsl_mask)
        fsl_mask_ims[:, 1, ...] = np.copy(fsl_mask)
        fsl_mask_ims[:, 2, ...] = np.copy(fsl_mask)

        fsl_reshape, _ = resample_isotropic(args, fsl_mask_ims, metadata)[:, 0, ...]
        fsl_reshape = (fsl_reshape >= 0.5).astype(fsl_mask.dtype)
    return fsl_reshape

def apply_preprocess(ims, unmasked_ims, metadata):
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
    if not args.pad_for_size or ims.shape[2] >= args.pad_for_size:
        return ims, metadata

    ims_pad = sup.zero_pad(ims, target_size=args.pad_for_size)
    print('Shape after zero padding...', ims_pad.shape)
    metadata['zero_pad_size'] = (ims_pad.shape[2], ims_pad.shape[3])

    return ims_pad, metadata

def preprocess_chain(args):
    metadata = {
        'lambda': []
    }

    ims, hdr, metadata = get_images(args, metadata)
    unmasked_ims = np.copy(ims)

    # first apply a mask to remove regions with zero signal
    ims, mask, metadata = mask_images(args, ims, metadata)

    # next apply a BET mask to remove non-brain tissue
    fsl_mask = fsl_brain_mask(args)
    ims = apply_fsl_mask(args, ims, fsl_mask)

    # scale and register images based on BET images
    ims, metadata = dicom_scaling(args, ims, hdr, metadata)
    ims, metadata = register(args, ims, metadata)
    ims, metadata = hist_norm(args, ims, metadata)
    ims, metadata = zoom_process(args, ims, metadata)

    ims, ims_mod, metadata = prescale_process(args, ims, mask, metadata)
    ims, ims_mod, metadata = match_scales(args, ims, ims_mod, metadata)
    ims, metadata = global_norm(args, ims, ims_mod, metadata)

    # reapply the preprocessing chain to the unmasked images
    unmasked_ims, metadata = apply_preprocess(ims, unmasked_ims, metadata)

    ims, _ = resample_isotropic(args, ims, metadata) # dont save to metadata on masked ims
    unmasked_ims, metadata = resample_isotropic(args, unmasked_ims, metadata)
    fsl_mask = reshape_fsl_mask(args, fsl_mask, metadata)

    ims, metadata = fsl_reject_slices(args, ims, fsl_mask, metadata)

    ims, _ = zero_pad(args, ims, metadata) # dont save to metadata on masked ims
    unmasked_ims, metadata = zero_pad(args, unmasked_ims, metadata)

    return unmasked_ims, ims, metadata

if __name__ == '__main__':
    args = fetch_args()
    ims, ims_mask, metadata = preprocess_chain(args)
    suio.save_data_h5(args.out_file, data=ims, data_mask=ims_mask, h5_key='data', metadata=metadata)
