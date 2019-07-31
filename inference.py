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

import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
import sigpy as sp

import keras.callbacks

import subtle.subtle_dnn as sudnn
import subtle.subtle_io as suio
import subtle.subtle_generator as sugen
import subtle.subtle_loss as suloss
import subtle.subtle_plot as suplot
import subtle.subtle_preprocess as supre
import subtle.subtle_metrics as sumetrics

from preprocess import preprocess_chain

import subtle.subtle_args as sargs

usage_str = 'usage: %(prog)s [options]'
description_str = 'Run SubtleGrad inference on dicom data'

if __name__ == '__main__':


    parser = sargs.parser(usage_str, description_str)
    args = parser.parse_args()

    print(args)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())


    if args.gpu_device is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device


    if args.data_preprocess:
        if args.verbose:
            print('loading preprocessed data from', args.data_preprocess)
        data = suio.load_file(args.data_preprocess)
        metadata = suio.load_h5_metadata(args.data_preprocess)
        args.path_zero, args.path_low, args.path_full = suio.get_dicom_dirs(args.path_base, override=args.override)
    else:
        if args.verbose:
            print('pre-processing data')
        data, metadata = preprocess_chain(args)
        if args.verbose:
            print('done')


    # get ground-truth for testing (e.g. hist re-normalization)
    im_gt, hdr_gt = suio.dicom_files(args.path_full, normalize=False)

    if args.denoise:
        if args.verbose:
            print('Denoise mode')
        data[:,1,:,:] = data[:,0,:,:].copy()

    original_data = np.copy(data)

    if args.resample_size is not None:
        print('Resampling data to {}'.format(args.resample_size))
        data = supre.resample_slices(data, resample_size=args.resample_size)

    ns, _, nx, ny = data.shape

    sudnn.clear_keras_memory()
    sudnn.set_keras_memory(args.keras_memory)

    loss_function = suloss.mixed_loss(l1_lambda=args.l1_lambda, ssim_lambda=args.ssim_lambda)
    metrics_monitor = [suloss.l1_loss, suloss.ssim_loss, suloss.mse_loss]
    m = sudnn.DeepEncoderDecoder2D(
            num_channel_input=2 * args.slices_per_input, num_channel_output=1,
            img_rows=nx, img_cols=ny,
            num_channel_first=args.num_channel_first,
            loss_function=loss_function,
            metrics_monitor=metrics_monitor,
            lr_init=args.lr_init,
            batch_norm=args.batch_norm,
            verbose=args.verbose,
            checkpoint_file=args.checkpoint_file,
            job_id=args.job_id)

    m.load_weights()


    # FIXME: change generator to work with ndarray directly, so that we don't have to write the inputs to disk


    tic = time.time()

    print('predicting...')

    if args.verbose:
        print(args.path_base)

    if args.inference_mpr:
        if args.verbose:
            print('running inference on orthogonal planes')
        slice_axes = [0, 2, 3]
    else:
        slice_axes = [args.slice_axis]

    checkpoint_files = [args.checkpoint_file_0, args.checkpoint_file_2, args.checkpoint_file_3]

    Y_predictions = np.zeros((ns, nx, ny, len(slice_axes), args.num_rotations))

    if args.inference_mpr and args.num_rotations > 1:
        angles = np.linspace(0, 90, args.num_rotations, endpoint=False)
    else:
        angles = [0]

    for rr, angle in enumerate(angles):
        if args.num_rotations > 1 and angle > 0:
            if args.verbose:
                print('{}/{} rotating by {} degrees'.format(rr+1, args.num_rotations, angle))
            data_rot = rotate(data, angle, reshape=False, axes=(0, 2))
        else:
            data_rot = data

        with tempfile.TemporaryDirectory() as tmpdirname:

            data_file = '{}/data.h5'.format(tmpdirname)
            suio.save_data_h5(data_file, data=data_rot, h5_key='data', metadata=metadata)

            for ii, slice_axis in enumerate(slice_axes):

                prediction_generator = sugen.DataGenerator(data_list=[data_file],
                        batch_size=1,
                        shuffle=False,
                        verbose=args.verbose,
                        residual_mode=args.residual_mode,
                        slices_per_input=args.slices_per_input,
                        resize=args.resize,
                        slice_axis=slice_axis)

                if checkpoint_files[ii]:
                    m.load_weights(checkpoint_files[ii])

                _Y_prediction = m.model.predict_generator(generator=prediction_generator, max_queue_size=args.max_queue_size, workers=args.num_workers, use_multiprocessing=args.use_multiprocessing, verbose=args.verbose)

                if slice_axis == 0:
                    pass
                elif slice_axis == 2:
                    _Y_prediction = np.transpose(_Y_prediction, (1, 0, 2, 3))
                elif slice_axis == 3:
                    _Y_prediction = np.transpose(_Y_prediction, (1, 2, 0, 3))

                if args.resize:
                    _Y_prediction = sp.util.resize(_Y_prediction, [ns, nx, ny, 1])

                if args.num_rotations > 1 and angle > 0:
                    _Y_prediction = rotate(_Y_prediction, -angle, reshape=False, axes=(0, 1))


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


    data = data.transpose((0, 2, 3, 1))
    original_data = original_data.transpose((0, 2, 3, 1))

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

    if args.predict_dir:
        # save raw data
        data_file_base = os.path.basename(data_file)
        _1, _2 = os.path.splitext(data_file_base)
        data_file_predict = '{}/{}_predict_{}.{}'.format(args.predict_dir, _1, args.job_id, args.predict_file_ext)
        suio.save_data(data_file_predict, Y_prediction, file_type=args.predict_file_ext)

    data_out = supre.undo_scaling(Y_prediction, metadata, verbose=args.verbose, im_gt=im_gt)
    suio.write_dicoms(args.path_zero, data_out, args.path_out, series_desc_pre='SubtleGad: ', series_desc_post=args.description, series_num=args.series_num)

    if args.stats_file:
        print('running stats on inference...')
        stats = {'pred/nrmse': [], 'pred/psnr': [], 'pred/ssim': [], 'low/nrmse': [], 'low/psnr': [], 'low/ssim': []}


        x_zero = original_data[...,0].squeeze()
        x_low = original_data[...,1].squeeze()
        x_full = original_data[...,2].squeeze()
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
