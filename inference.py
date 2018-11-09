
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

parser.add_argument('--description', action='store', dest='description', type=str, help='append to end of series description', default='')
parser.add_argument('--path_out', action='store', dest='path_out', type=str, help='path to output SubtleGad dicom dir', default=None)
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

parser.add_argument('--data_list', action='store', dest='data_list_file', type=str, help='list of pre-processed files for training', default=None)
parser.add_argument('--data_dir', action='store', dest='data_dir', type=str, help='location of data', default=None)
parser.add_argument('--file_ext', action='store', dest='file_ext', type=str, help='file extension of data', default=None)
parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs to run', default=10)
parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, help='batch size', default=8)
parser.add_argument('--tbimage_batch_size', action='store', dest='tbimage_batch_size', type=int, help='TBImage batch size', default=8)
parser.add_argument('--gpu', action='store', dest='gpu_device', type=str, help='set GPU', default=None)
parser.add_argument('--keras_memory', action='store', dest='keras_memory', type=float, help='set Keras memory (0 to 1)', default=1.)
parser.add_argument('--checkpoint', action='store', dest='checkpoint_file', type=str, help='checkpoint file', default=None)
parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
parser.add_argument('--validation_split', action='store', dest='validation_split', type=float, help='ratio of validation data', default=.1)
parser.add_argument('--log_dir', action='store', dest='log_dir', type=str, help='log directory', default='logs')
parser.add_argument('--max_data_sets', action='store', dest='max_data_sets', type=int, help='limit number of data sets', default=None)
parser.add_argument('--predict', action='store', dest='predict_dir', type=str, help='perform prediction and write to directory', default=None)
parser.add_argument('--learn_residual', action='store_true', dest='residual_mode', help='learn residual, (zero, low - zero, full - zero)', default=False)
parser.add_argument('--learning_rate', action='store', dest='lr_init', type=float, help='intial learning rate', default=.001)
parser.add_argument('--batch_norm', action='store_true', dest='batch_norm', help='batch normalization')
parser.add_argument('--steps_per_epoch', action='store', dest='steps_per_epoch', type=int, help='number of iterations per epoch (default -- # slices in dataset / batch_size', default=None)
parser.add_argument('--val_steps_per_epoch', action='store', dest='val_steps_per_epoch', type=int, help='# slices in dataset / batch_size', default=None)
parser.add_argument('--use_multiprocessing', action='store_true', dest='use_multiprocessing', help='use multiprocessing in generator', default=False)
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, help='number of workers for generator', default=1)
parser.add_argument('--max_queue_size', action='store', dest='max_queue_size', type=int, help='generator queue size', default=16)
parser.add_argument('--shuffle', action='store_true', dest='shuffle', help='shuffle input data files each epoch', default=False)
parser.add_argument('--history_file', action='store', dest='history_file', type=str, help='store history in npy file', default=None)
parser.add_argument('--id', action='store', dest='job_id', type=str, help='job id for logging', default='')
parser.add_argument('--slices_per_input', action='store', dest='slices_per_input', type=int, help='number of slices per input (2.5D)', default=1)
parser.add_argument('--predict_file_ext', action='store', dest='predict_file_ext', type=str, help='file extension of predcited data', default='npy')
parser.add_argument('--num_channel_first', action='store', dest='num_channel_first', type=int, help='first layer channels', default=32)
parser.add_argument('--gen_type', action='store', dest='gen_type', type=str, help='generator type (legacy or split)', default='legacy')
parser.add_argument('--ssim_lambda', action='store', type=float, dest='ssim_lambda', help='include ssim loss with weight ssim_lambda', default=0.)
parser.add_argument('--l1_lambda', action='store', type=float, dest='l1_lambda', help='include L1 loss with weight l1_lambda', default=1.)


if __name__ == '__main__':


    args = parser.parse_args()

    print(args)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())


    if args.gpu_device is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device


    data, metadata = preprocess_chain(args)
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
                #log_dir=log_tb_dir,
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
                #log_dir=log_tb_dir,
                job_id=args.job_id)

    m.load_weights()


    # FIXME: change generator to work with ndarray directly, so that we don't have to write the inputs to disk
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_file = '{}/data.h5'.format(tmpdirname)
        suio.save_data_h5(data_file, data=data, h5_key='data', metadata=metadata)


        tic = time.time()
        if args.predict_dir is not None:

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



            #data_file_base = os.path.basename(data_file)
            #_1, _2 = os.path.splitext(data_file_base)
            #data_file_predict = '{}/{}_predict_{}.{}'.format(args.predict_dir, _1, args.job_id, args.predict_file_ext)

            #if args.verbose:
                #print('output: {}'.format(data_file_predict))

            #suio.save_data(data_file_predict, Y_prediction, file_type=args.predict_file_ext)
            #for __idx in np.linspace(.1*Y_prediction.shape[0], .9*Y_prediction.shape[0], 5):
                #_idx = int(__idx)
                #plot_file_predict = '{}/plots/{}_predict_{}_{:03d}.png'.format(args.predict_dir, _1, args.job_id, _idx)
                #suplot.compare_output(data.transpose((0, 3, 1, 2)), Y_prediction, idx=_idx, show_diff=False, output=plot_file_predict)

        toc = time.time()
        print('done predicting ({:.0f} sec)'.format(toc - tic))

