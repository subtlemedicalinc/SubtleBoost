
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

usage_str = 'usage: %(prog)s [options]'
description_str = 'Run SubtleGrad inference on dicom data'

parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', is_config_file=True, help='config file path', default=False)

parser.add_argument('--data_preprocess', action='store', dest='data_preprocess', type=str, help='load already-preprocessed data', default=False)
parser.add_argument('--description', action='store', dest='description', type=str, help='append to end of series description', default='')
parser.add_argument('--path_out', action='store', dest='path_out', type=str, help='path to output SubtleGad dicom dir', default=None)
parser.add_argument('--path_base', action='store', dest='path_base', type=str, help='path to base dicom directory containing subdirs', default=None)
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

parser.add_argument('--gpu', action='store', dest='gpu_device', type=str, help='set GPU', default=None)
parser.add_argument('--keras_memory', action='store', dest='keras_memory', type=float, help='set Keras memory (0 to 1)', default=1.)
parser.add_argument('--checkpoint', action='store', dest='checkpoint_file', type=str, help='checkpoint file', default=None)
parser.add_argument('--learn_residual', action='store_true', dest='residual_mode', help='learn residual, (zero, low - zero, full - zero)', default=False)
parser.add_argument('--learning_rate', action='store', dest='lr_init', type=float, help='intial learning rate', default=.001)
parser.add_argument('--batch_norm', action='store_true', dest='batch_norm', help='batch normalization')
parser.add_argument('--use_multiprocessing', action='store_true', dest='use_multiprocessing', help='use multiprocessing in generator', default=False)
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, help='number of workers for generator', default=1)
parser.add_argument('--max_queue_size', action='store', dest='max_queue_size', type=int, help='generator queue size', default=16)
parser.add_argument('--id', action='store', dest='job_id', type=str, help='job id for logging', default='')
parser.add_argument('--slices_per_input', action='store', dest='slices_per_input', type=int, help='number of slices per input (2.5D)', default=1)
parser.add_argument('--num_channel_first', action='store', dest='num_channel_first', type=int, help='first layer channels', default=32)
parser.add_argument('--gen_type', action='store', dest='gen_type', type=str, help='generator type (legacy or split)', default='legacy')
parser.add_argument('--ssim_lambda', action='store', type=float, dest='ssim_lambda', help='include ssim loss with weight ssim_lambda', default=0.)
parser.add_argument('--l1_lambda', action='store', type=float, dest='l1_lambda', help='include L1 loss with weight l1_lambda', default=1.)


if __name__ == '__main__':


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

