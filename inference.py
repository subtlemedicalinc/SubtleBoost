#!/usr/bin/env python
'''
inference.py

Inference for contrast synthesis.
Runs the full inference pipeline on a patient

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/11/09
'''


import sys

import tempfile
import os
import datetime
import time
import random
from warnings import warn
import configargparse as argparse

import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_fill_holes
import sigpy as sp

import keras.callbacks

from subtle.dnn.helpers import clear_keras_memory, set_keras_memory, load_model, load_data_loader, gan_model
import subtle.utils.io as utils_io
import subtle.utils.experiment as utils_exp
from scipy.ndimage.interpolation import rotate
import subtle.subtle_loss as suloss
import subtle.subtle_plot as suplot
import subtle.subtle_preprocess as supre
import subtle.subtle_metrics as sumetrics

from preprocess import preprocess_chain

import subtle.subtle_args as sargs

usage_str = 'usage: %(prog)s [options]'
description_str = 'Run SubtleGrad inference on dicom data'

def save_img(img, fname):

    import matplotlib.pyplot as plt
    plt.set_cmap('gray')

    plt.imshow(img)
    plt.colorbar()
    plt.savefig('/home/srivathsa/projects/studies/gad/tiantan/inference/test/{}.png'.format(fname))
    plt.clf()

def resample_unisotropic(args, ims, metadata):
    print('undoing resample isotropic...')
    res_iso = [args.resample_isotropic] * 3

    uniso_zero, _ = supre.zoom_iso(ims[:, 0, ...], res_iso, metadata['old_spacing_zero'])
    uniso_low, _ = supre.zoom_iso(ims[:, 1, ...], res_iso, metadata['old_spacing_low'])
    uniso_full, _ = supre.zoom_iso(ims[:, 2, ...], res_iso, metadata['old_spacing_full'])

    data_uniso = np.array([uniso_zero, uniso_low, uniso_full]).transpose(1, 0, 2, 3)
    print('Data after undoing isotropic', data_uniso.shape)

    return data_uniso

def inference_process(args):
    print('------')
    print(args.debug_print())
    print('------\n\n\n')

    args.gpu = str(args.gpu)

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.dicom_inference:
        if args.verbose:
            print('loading preprocessed data from', args.data_preprocess)

        data = utils_io.load_file(args.data_preprocess, params={'h5_key': 'data'})
        data_mask = utils_io.load_file(args.data_preprocess, params={'h5_key': 'data_mask'}) if utils_io.has_h5_key(args.data_preprocess, 'data_mask') else None

        metadata = utils_io.load_h5_metadata(args.data_preprocess)
        dicom_dirs = utils_io.get_dicom_dirs(args.path_base, override=args.override)

        args.path_zero = dicom_dirs[0]
        args.path_low = dicom_dirs[1]

        if len(dicom_dirs) == 3:
            args.path_full = dicom_dirs[2]
            metadata['inference_only'] = False
        else:
            args.path_full = args.path_low
            metadata['inference_only'] = True
    else:
        if args.verbose:
            print('pre-processing data')
        data, data_mask, metadata = preprocess_chain(args)
        if args.verbose:
            print('done')

    # get ground-truth for testing (e.g. hist re-normalization)
    im_gt, hdr_gt = utils_io.dicom_files(args.path_full, normalize=False)

    if args.denoise:
        if args.verbose:
            print('Denoise mode')
        data[:,1,:,:] = data[:,0,:,:].copy()
        if data_mask is not None:
            data_mask[:,1,:,:] = data_mask[:,0,:,:].copy()

    original_data = np.copy(data)
    original_data_mask = np.copy(data_mask) if data_mask is not None else None

    if args.resample_size is not None:
        print('Resampling data to {}'.format(args.resample_size))
        data = supre.resample_slices(data, resample_size=args.resample_size)
        data_mask = supre.resample_slices(data_mask, resample_size=args.resample_size)

    # Center position
    if not args.brain_only and args.brain_centering:
        bbox_arr = []
        data_mod = []
        for cont in np.arange(data.shape[1]):
            data_cont, bbox = supre.center_position_brain(data[:, cont, ...], threshold=0.1)
            data_mod.append(data_cont)
            bbox_arr.append(bbox)

        data = np.array(data_mod).transpose(1, 0, 2, 3)

    ns, _, nx, ny = data.shape

    clear_keras_memory()
    set_keras_memory(args.keras_memory)

    loss_function = suloss.mixed_loss(l1_lambda=args.l1_lambda, ssim_lambda=args.ssim_lambda)
    metrics_monitor = [suloss.l1_loss, suloss.ssim_loss, suloss.mse_loss]

    model_class = load_model(args.model_name)

    model_kwargs = {
        'model_config': args.model_config,
        'num_channel_output': 1,
        'loss_function': loss_function,
        'metrics_monitor': metrics_monitor,
        'lr_init': args.lr_init,
        'verbose': args.verbose,
        'checkpoint_file': args.checkpoint_file,
        'job_id': args.job_id
    }

    # FIXME: change generator to work with ndarray directly, so that we don't have to write the inputs to disk

    tic = time.time()

    print('predicting...')

    if args.verbose:
        print(args.path_base)

    gen_kwargs = {
        'batch_size': 1,
        'shuffle': False,
        'verbose': args.verbose
    }
    predict_kwargs = {
        'max_queue_size': args.max_queue_size,
        'workers': args.num_workers,
        'use_multiprocessing': args.use_multiprocessing,
        'verbose': args.verbose
    }
    data_loader = load_data_loader(args.model_name)

    mconf_dict = utils_exp.get_model_config(args.model_name, args.model_config, model_type='generators')
    num_poolings = mconf_dict['num_poolings'] if 'num_poolings' in mconf_dict else 3

    ### Start IF condition for 3D patch based
    if '3d' in args.model_name:
        data_pad = supre.zero_pad_for_dnn(data, num_poolings=num_poolings)
        data_pad_mask = supre.zero_pad_for_dnn(data_mask, num_poolings=num_poolings)
        ns, _, nx, ny = data_pad.shape

        kw_model = {
            'img_rows': args.block_size if not args.predict_full_volume else nx,
            'img_cols': args.block_size if not args.predict_full_volume else ny,
            'img_depth': args.block_size if not args.predict_full_volume else ns,
            'num_channel_input': 2
        }
        model_kwargs = {**model_kwargs, **kw_model}

        m = model_class(**model_kwargs)
        m.load_weights()

        kw = {
            'data_list': [args.data_preprocess],
            'predict': True,
            'block_size': args.block_size,
            'block_strides': args.block_strides,
            'batch_size': args.batch_size,
            'predict_full': args.predict_full_volume
        }
        gen_kwargs = {**gen_kwargs, **kw}

        original_data = data
        original_data_mask = data_mask

        prediction_generator = data_loader(**gen_kwargs)

        predict_kwargs['generator'] = prediction_generator

        if args.predict_full_volume:
            prediction_generator._cache_img(args.data_preprocess, ims=data_pad, ims_mask=data_pad_mask)

            predict_kwargs['max_queue_size'] = 1
            x_pred, _, _ = prediction_generator.__getitem__(0)
            Y_prediction = m.model.predict(x_pred, batch_size=1, verbose=args.verbose)
            Y_prediction = Y_prediction[0, ...].transpose(2, 0, 1, 3)
            y_pred = supre.center_crop(Y_prediction[..., 0], data[:, 0, ...])
        else:
            prediction_generator._cache_img(args.data_preprocess, ims=data_pad.transpose(1, 0, 2, 3), ims_mask=data_pad_mask.transpose(1, 0, 2, 3))

            Y_prediction = []
            for idx in range(prediction_generator.__len__()):
                x_pred, _, _ = prediction_generator.__getitem__(idx)

                Y_prediction.extend(m.model.predict(x_pred, batch_size=1, verbose=args.verbose))

            Y_prediction = np.array(Y_prediction)

            n_chunks = lambda sh_idx: ((data[:, 0, ...].shape[sh_idx] - args.block_size + args.block_strides) // args.block_strides) + 1

            bshape = int(np.cbrt(Y_prediction.shape[0]))

            Y_prediction = sp.blocks_to_array(
                input=Y_prediction.reshape(n_chunks(0), n_chunks(1), n_chunks(2), args.block_size, args.block_size, args.block_size),
                oshape=(ns, nx, ny),
                blk_shape=[args.block_size]*3,
                blk_strides=[args.block_strides]*3
            )

            y_pred = supre.center_crop(Y_prediction, data[:, 0, ...])
        Y_prediction = np.array([y_pred]).transpose(1, 2, 3, 0)
    else:
        kw_model = {
            'img_rows': nx,
            'img_cols': ny,
            'num_channel_input': 2 * args.slices_per_input
        }
        model_kwargs = {**model_kwargs, **kw_model}

        m = model_class(**model_kwargs)
        m.load_weights()

        kw = {
            'residual_mode': args.residual_mode,
            'slices_per_input': args.slices_per_input,
            'resize': args.resize,
            'brain_only': args.brain_only
        }
        gen_kwargs = {**gen_kwargs, **kw}

        if args.inference_mpr:
            if args.verbose:
                print('running inference on orthogonal planes')
            slice_axes = [0, 2, 3]
        else:
            slice_axes = [args.slice_axis]

        if args.gan_mode:
            gen = m.model
            d = load_model(args.adversary_name)(img_rows=nx, img_cols=ny, compile_model=True)
            disc = d.model

            gan = gan_model(gen, disc, (nx, ny, 2 * args.slices_per_input))
            gan.load_weights(args.checkpoint_file)

        Y_predictions = np.zeros((ns, nx, ny, len(slice_axes), args.num_rotations))

        if args.inference_mpr and args.num_rotations > 1:
            angles = np.linspace(0, 90, args.num_rotations, endpoint=False)
        else:
            angles = [0]

        for rr, angle in enumerate(angles):
            if args.num_rotations > 1 and angle > 0:
                if args.verbose:
                    print('{}/{} rotating by {} degrees'.format(rr+1, args.num_rotations, angle))
                data_rot = rotate(data, angle, reshape=args.reshape_for_mpr_rotate, axes=(0, 2))
                data_rot = supre.zero_pad_for_dnn(data_rot, num_poolings=num_poolings)

                if data_mask is not None:
                    data_mask_rot = rotate(data_mask, angle, reshape=args.reshape_for_mpr_rotate, axes=(0, 2))
                    data_mask_rot = supre.zero_pad_for_dnn(data_mask_rot, num_poolings=num_poolings)
                else:
                    data_mask_rot = None
            else:
                data_rot = supre.zero_pad_for_dnn(data, num_poolings=num_poolings)
                if data_mask is not None:
                    data_mask_rot = supre.zero_pad_for_dnn(data_mask, num_poolings=num_poolings)
                else:
                    data_mask_rot = None

            with tempfile.TemporaryDirectory() as tmpdirname:
                data_file = '{}/data.h5'.format(tmpdirname)
                original_scale = (data_rot.min(), data_rot.max())

                if args.match_scales_fsl:
                    data_rot = np.interp(data_rot, original_scale, (data_mask_rot.min(), data_mask_rot.max()))
                    print(data_rot.min(), data_rot.max())

                params = {
                    'metadata': metadata,
                    'data': data_rot,
                    'data_mask': data_mask_rot,
                    'h5_key': 'data'
                }
                utils_io.save_data_h5(data_file, **params)

                for ii, slice_axis in enumerate(slice_axes):
                    gen_kwargs['data_list'] = [data_file]
                    gen_kwargs['slice_axis'] = [slice_axis]
                    prediction_generator = data_loader(**gen_kwargs)

                    data_ref = np.zeros_like(data_rot)
                    if slice_axis == 2:
                        data_ref = np.transpose(data_ref, (2, 1, 0, 3))
                    elif slice_axis == 3:
                        data_ref = np.transpose(data_ref, (3, 1, 0, 2))

                    if args.reshape_for_mpr_rotate:
                        m.img_rows = data_ref.shape[2]
                        m.img_cols = data_ref.shape[3]
                        m.verbose = False
                        m._build_model()
                        m.load_weights()

                    predict_kwargs['generator'] = prediction_generator
                    _Y_prediction = m.model.predict_generator(**predict_kwargs)
                    # N, x, y, 1

                    if args.match_scales_fsl:
                        _Y_prediction = np.interp(_Y_prediction, (_Y_prediction.min(), _Y_prediction.max()), original_scale)

                    if slice_axis == 0:
                        pass
                    elif slice_axis == 2:
                        _Y_prediction = np.transpose(_Y_prediction, (1, 0, 2, 3))
                    elif slice_axis == 3:
                        _Y_prediction = np.transpose(_Y_prediction, (1, 2, 0, 3))

                    if args.resize:
                        _Y_prediction = sp.util.resize(_Y_prediction, [ns, nx, ny, 1])

                    pred_rotated = False

                    if args.num_rotations > 1 and angle > 0:
                        pred_rotated = True
                        _Y_prediction = rotate(_Y_prediction, -angle, reshape=args.reshape_for_mpr_rotate, axes=(0, 1))

                    if pred_rotated or _Y_prediction.shape[0] != data.shape[0]:
                        y_pred = _Y_prediction[..., 0]
                        y_pred = supre.center_crop(y_pred, data[:, 0, ...])
                        _Y_prediction = np.array([y_pred]).transpose(1, 2, 3, 0)

                    Y_predictions[..., ii, rr] = _Y_prediction[..., 0]

        if args.verbose and args.inference_mpr:
            print('averaging each plane')

        if 'mean' in args.inference_mpr_avg:
            Y_masks_sum = np.sum(np.array(Y_predictions > 0, dtype=np.float), axis=(-1, -2), keepdims=False)
            Y_prediction = np.divide(np.sum(Y_predictions, axis=(-1, -2), keepdims=False), Y_masks_sum, where=Y_masks_sum > 0)[..., None]
        elif 'median' in args.inference_mpr_avg:
            assert args.num_rotations == 1, 'Median not currently supported when doing multiple rotations (need to account for non-zeros only)'
            # FIXME need to apply median along non-zero values only. Something like this:
            #out = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, x123)
            Y_prediction = np.median(Y_predictions, axis=[-1, -2], keepdims=False)[..., None]

    # End IF for 3D patch based


    # if 'zero_pad_size' in metadata:
    if args.undo_pad_resample:
        splits = args.undo_pad_resample.split(',')
        crop_x, crop_y = int(splits[0]), int(splits[1])

        y_pred = Y_prediction[..., 0]
        ref_img = np.zeros((y_pred.shape[0], crop_x, crop_y))
        y_pred = supre.center_crop(y_pred, ref_img)
        Y_prediction = np.array([y_pred]).transpose(1, 2, 3, 0)

        od_crop = np.zeros((y_pred.shape[0], 3, crop_x, crop_y))
        od_mask_crop = np.zeros((y_pred.shape[0], 3, crop_x, crop_y))
        for c in np.arange(3):
            od_crop[:, c, ...] = supre.center_crop(original_data[:, c, ...], ref_img)
            od_mask_crop[:, c, ...] = supre.center_crop(original_data_mask[:, c, ...], ref_img)

        original_data = od_crop
        original_data_mask = od_mask_crop

        print('Y prediction shape after undoing zero pad', Y_prediction.shape)


        # isotropic resampling has been done in preprocess, so need to unresample to original spacing
        # res_iso = [args.resample_isotropic] * 3
        # y_pred, _ = supre.zoom_iso(Y_prediction[..., 0], res_iso, metadata['old_spacing_zero'])
        # Y_prediction = np.array([y_pred]).transpose(1, 2, 3, 0)
        #
        # original_data = resample_unisotropic(args, original_data, metadata)
        # original_data_mask = resample_unisotropic(args, original_data_mask, metadata)

    data = data.transpose((0, 2, 3, 1))
    original_data = original_data.transpose((0, 2, 3, 1))

    if np.any(original_data_mask):
        original_data_mask = original_data_mask.transpose((0, 2, 3, 1))

    # if residual mode is on, we need to add the original contrast back in
    if args.residual_mode:
        h = args.slices_per_input // 2
        Y_prediction = data[:,:,:,0].squeeze() + Y_prediction.squeeze()

    if args.zoom:
        data_shape = metadata['zoom_dims']
        if args.verbose:
            print('unzoom')
            ticz = time.time()
        Y_prediction = zoom(Y_prediction[...,0], zoom=(1, data_shape[2]/args.zoom, data_shape[3]/args.zoom), order=args.zoom_order)[...,None]
        if args.verbose:
            tocz = time.time()
            print('unzoom done: {} s'.format(tocz-ticz))

    if args.resample_size and original_data.shape[2] != args.resample_size:
        Y_prediction = np.transpose(Y_prediction, (0, 3, 1, 2))
        Y_prediction = supre.resample_slices(Y_prediction, resample_size=original_data.shape[2])
        Y_prediction = np.transpose(Y_prediction, (0, 2, 3, 1))

    # undo brain center

    if not args.brain_only and args.brain_centering:
        y_pred = Y_prediction[..., 0]
        y_pred_cont = supre.undo_brain_center(y_pred, bbox_arr[0], threshold=0.1)
        Y_prediction = np.array([y_pred_cont]).transpose(1, 2, 3, 0)

    # if 'zero_pad_size' in metadata:
    #     if 'resampled_size' in metadata:
    #         crop_size = metadata['resampled_size'][0]
    #     else:
    #         crop_size = metadata['original_size'][0]
    #
    #     y_pred = Y_prediction[..., 0]
    #     ref_img = np.zeros((y_pred.shape[0], crop_size, crop_size))
    #     y_pred = supre.center_crop(y_pred, ref_img)
    #     Y_prediction = np.array([y_pred]).transpose(1, 2, 3, 0)
    #
    #     print('Y prediction shape after undoing zero pad', Y_prediction.shape)

    if args.predict_dir:
        # save raw data
        data_file_base = os.path.basename(data_file)
        _1, _2 = os.path.splitext(data_file_base)
        data_file_predict = '{}/{}_predict_{}.{}'.format(args.predict_dir, _1, args.job_id, args.predict_file_ext)
        utils_io.save_data(data_file_predict, Y_prediction, file_type=args.predict_file_ext)

    data_out = supre.undo_scaling(Y_prediction, metadata, verbose=args.verbose, im_gt=im_gt)

    utils_io.write_dicoms(args.path_zero, data_out, args.path_out, series_desc_pre='SubtleGad: ', series_desc_post=args.description, series_num=args.series_num)

    if args.brain_only:
        data_zero = original_data[..., 0]
        brain_mask = binary_fill_holes(original_data_mask[..., 0] > 0.1)

        y_pred = (data_zero - (data_zero * brain_mask)) + Y_prediction[..., 0]
        Y_prediction_stitch = np.array([y_pred]).transpose(1, 2, 3, 0)

        data_out_stitch = supre.undo_scaling(Y_prediction_stitch, metadata, verbose=args.verbose, im_gt=im_gt)

        utils_io.write_dicoms(args.path_zero, data_out_stitch, args.path_out + '_stitch', series_desc_pre='SubtleGad: ', series_desc_post=args.description + '_stitch', series_num=args.series_num)

    if args.stats_file and not metadata['inference_only']:
        print('running stats on inference...')
        stats = {'pred/nrmse': [], 'pred/psnr': [], 'pred/ssim': [], 'low/nrmse': [], 'low/psnr': [], 'low/ssim': []}

        data_metrics = original_data_mask if args.brain_only else original_data

        x_zero = data_metrics[...,0].squeeze()
        x_low = data_metrics[...,1].squeeze()
        x_full = data_metrics[...,2].squeeze()
        x_pred = Y_prediction.squeeze().astype(np.float32)

        stats['low/nrmse'].append(sumetrics.nrmse(x_full, x_low))
        stats['low/ssim'].append(sumetrics.ssim(x_full, x_low))
        stats['low/psnr'].append(sumetrics.psnr(x_full, x_low))

        stats['pred/nrmse'].append(sumetrics.nrmse(x_full, x_pred))
        stats['pred/ssim'].append(sumetrics.ssim(x_full, x_pred))
        stats['pred/psnr'].append(sumetrics.psnr(x_full, x_pred))

        if args.verbose:
            for key in stats.keys():
                print('{}: {}'.format(key, stats[key]))

        print('Saving stats to {}'.format(args.stats_file))
        with h5py.File(args.stats_file, 'w') as f:
            for key in stats.keys():
                f.create_dataset(key, data=stats[key])


    toc = time.time()
    print('done predicting ({:.0f} sec)'.format(toc - tic))
