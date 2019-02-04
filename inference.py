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

import numpy as np

import keras.callbacks

import subtle.subtle_dnn as sudnn
import subtle.subtle_io as suio
import subtle.subtle_generator as sugen
import subtle.subtle_loss as suloss
import subtle.subtle_plot as suplot
import subtle.subtle_preprocess as supre

from preprocess import preprocess_chain

import subtle.subtle_args as sargs

usage_str = 'usage: %(prog)s [options]'
description_str = 'Run SubtleGrad inference on dicom data'

if __name__ == '__main__':


    parser = sargs.parser(usage_str, description_str)
    args = parser.parse_args()

    args.path_zero = None
    args.path_low = None
    args.path_full = None

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
        args.path_zero, args.path_low, args.path_full = suio.get_dicom_dirs(args.path_base)
    else:
        if args.verbose:
            print('pre-processing data')
        data, metadata = preprocess_chain(args)
        if args.verbose:
            print('done')

    ns, _, nx, ny = data.shape

    sudnn.clear_keras_memory()
    sudnn.set_keras_memory(args.keras_memory)

    loss_function = suloss.mixed_loss(l1_lambda=args.l1_lambda, ssim_lambda=args.ssim_lambda)
    metrics_monitor = [suloss.l1_loss, suloss.ssim_loss, suloss.mse_loss]
    if args.gen_type == 'legacy':
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

    elif args.gen_type == 'split':
        m = sudnn.DeepEncoderDecoder2D(
                num_channel_input=nz, num_channel_output=1,
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
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_file = '{}/data.h5'.format(tmpdirname)
        suio.save_data_h5(data_file, data=data, h5_key='data', metadata=metadata)


        tic = time.time()

        print('predicting...')


        if args.verbose:
            print(args.path_base)

        # use generator to maintain consistent data formatting
        prediction_generator = sugen.DataGenerator(data_list=[data_file],
                batch_size=1,
                shuffle=False,
                verbose=args.verbose, 
                residual_mode=args.residual_mode,
                slices_per_input=args.slices_per_input)

        Y_prediction = m.model.predict_generator(generator=prediction_generator, max_queue_size=args.max_queue_size, workers=args.num_workers, use_multiprocessing=args.use_multiprocessing, verbose=args.verbose)

        data = data.transpose((0, 2, 3, 1))

        # if residual mode is on, we need to add the original contrast back in
        if args.residual_mode:
            h = args.slices_per_input // 2
            Y_prediction = data[:,:,:,0].squeeze() + Y_prediction.squeeze()

        ## HERE
        data_out = supre.undo_scaling(Y_prediction, metadata, verbose=args.verbose)
        suio.write_dicoms(args.path_zero, data_out, args.path_out, series_desc_pre='SubtleGad: ', series_desc_post=args.description)

    toc = time.time()
    print('done predicting ({:.0f} sec)'.format(toc - tic))
