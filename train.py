#!/usr/bin/env python
'''
train.py

Training for contrast synthesis.
Trains netowrk using dataset of npy files

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/25
'''


import sys

print('------')
print(' '.join(sys.argv))
print('------\n\n\n')

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
import subtle.subtle_args as sargs

usage_str = 'usage: %(prog)s [options]'
description_str = 'Train SubtleGrad network on pre-processed data.'

# FIXME: add time stamps, logging
# FIXME: data augmentation


if __name__ == '__main__':

    parser = sargs.parser(usage_str, description_str)
    args = parser.parse_args()

    print(args)
    print("----------")
    print(parser.format_help())
    print("----------")
    print(parser.format_values())
    
    assert args.data_list_file is not None, 'must specify data list'


    if args.log_dir is not None:
        try:
            os.mkdir(args.log_dir)
        except Exception as e:
            warn(str(e))
            pass

    if args.predict_dir is not None:
        try:
            os.mkdir(args.predict_dir)
            os.mkdir('{}/plots'.format(args.predict_dir))
        except Exception as e:
            warn(str(e))
            pass

    if args.max_data_sets is None:
        max_data_sets = np.inf
    else:
        max_data_sets = args.max_data_sets

    if args.gpu_device is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    np.random.seed(args.random_seed)

    log_tb_dir = os.path.join(args.log_dir, '{}_{}'.format(args.job_id, time.time()))

    # load data
    if args.verbose:
        print('loading data from {}'.format(args.data_list_file))
    tic = time.time()

    data_list = suio.get_data_list(args.data_list_file, file_ext=args.file_ext, data_dir=args.data_dir)

    # each element of the data_list contains 3 sets of 3D
    # volumes containing zero, low, and full contrast.
    # the number of slices may differ but the image dimensions
    # should be the same

    # randomly grab max_data_sets from total data pool
    _ridx = np.random.permutation(len(data_list))
    data_list = [data_list[i] for i in _ridx[:args.max_data_sets]]

    # get dimensions from first file
    if args.gen_type == 'legacy':
        data_shape = suio.get_shape(data_list[0])
        _, _, nx, ny = data_shape
    elif args.gen_type == 'split':
        data_shape = suio.get_shape(data_list[0], params={'h5_key': 'data/X'})
        print(data_shape)
    #FIXME: check that image sizes are the same
        _, nx, ny, nz = data_shape

    sudnn.clear_keras_memory()
    sudnn.set_keras_memory(args.keras_memory)

    loss_function = suloss.mixed_loss(l1_lambda=args.l1_lambda, ssim_lambda=args.ssim_lambda)
    metrics_monitor = [suloss.l1_loss, suloss.ssim_loss, suloss.mse_loss, suloss.psnr_loss]

    m = sudnn.DeepEncoderDecoder2D(
            num_channel_input=len(args.input_idx) * args.slices_per_input, num_channel_output=len(args.output_idx),
            img_rows=nx, img_cols=ny,
            num_channel_first=args.num_channel_first,
            loss_function=loss_function,
            metrics_monitor=metrics_monitor,
            lr_init=args.lr_init,
            batch_norm=args.batch_norm,
            verbose=args.verbose,
            checkpoint_file=args.checkpoint_file,
            log_dir=log_tb_dir,
            job_id=args.job_id,
            save_best_only=args.save_best_only)

    m.load_weights()

    tic = time.time()
    print('training...')

    if len(data_list) == 1:
        r = 0
    elif args.validation_split == 0:
        r = 1
    else: # len(data_list) > 1
        r = int(len(data_list) * args.validation_split)

    data_val_list = data_list[:r]
    data_train_list = data_list[r:]

    if args.verbose:
        print('using {} datasets for training:'.format(len(data_train_list)))
        for d in data_train_list:
            print(d)
        print('using {} datasets for validation:'.format(len(data_val_list)))
        for d in data_val_list:
            print(d)


    callbacks = []
    callbacks.append(m.callback_checkpoint())
    callbacks.append(m.callback_tensorbaord(log_dir='{}_plot'.format(log_tb_dir)))
    if args.train_mpr:
        callbacks.append(m.callback_tbimage(data_list=data_val_list, slice_dict_list=None, slices_per_epoch=1, slices_per_input=args.slices_per_input, batch_size=args.tbimage_batch_size, verbose=args.verbose, residual_mode=args.residual_mode, tag='Validation Dir 1', gen_type=args.gen_type, log_dir='{}_image'.format(log_tb_dir), shuffle=True, input_idx=args.input_idx, output_idx=args.output_idx, slice_axis=0, resize=args.resize))
        callbacks.append(m.callback_tbimage(data_list=data_val_list, slice_dict_list=None, slices_per_epoch=1, slices_per_input=args.slices_per_input, batch_size=args.tbimage_batch_size, verbose=args.verbose, residual_mode=args.residual_mode, tag='Validation Dir 2', gen_type=args.gen_type, log_dir='{}_image'.format(log_tb_dir), shuffle=True, input_idx=args.input_idx, output_idx=args.output_idx, slice_axis=2, resize=args.resize))
        callbacks.append(m.callback_tbimage(data_list=data_val_list, slice_dict_list=None, slices_per_epoch=1, slices_per_input=args.slices_per_input, batch_size=args.tbimage_batch_size, verbose=args.verbose, residual_mode=args.residual_mode, tag='Validation Dir 3', gen_type=args.gen_type, log_dir='{}_image'.format(log_tb_dir), shuffle=True, input_idx=args.input_idx, output_idx=args.output_idx, slice_axis=3, resize=args.resize))
    else:
        callbacks.append(m.callback_tbimage(data_list=data_val_list, slice_dict_list=None, slices_per_epoch=1, slices_per_input=args.slices_per_input, batch_size=args.tbimage_batch_size, verbose=args.verbose, residual_mode=args.residual_mode, tag='Validation', gen_type=args.gen_type, log_dir='{}_image'.format(log_tb_dir), shuffle=True, input_idx=args.input_idx, output_idx=args.output_idx, slice_axis=args.slice_axis, resize=args.resize))
    #cb_tensorboard = m.callback_tensorbaord(log_every=1)


    if args.train_mpr:
        training_generator_0 = sugen.DataGenerator(data_list=data_train_list,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                verbose=args.verbose, 
                residual_mode=args.residual_mode,
                positive_only = args.positive_only,
                slices_per_input=args.slices_per_input,
                input_idx=args.input_idx,
                output_idx=args.output_idx,
                slice_axis=0,
                resize=args.resize)
        training_generator_2 = sugen.DataGenerator(data_list=data_train_list,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                verbose=args.verbose, 
                residual_mode=args.residual_mode,
                positive_only = args.positive_only,
                slices_per_input=args.slices_per_input,
                input_idx=args.input_idx,
                output_idx=args.output_idx,
                slice_axis=2,
                resize=args.resize)
        training_generator_3 = sugen.DataGenerator(data_list=data_train_list,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                verbose=args.verbose, 
                residual_mode=args.residual_mode,
                positive_only = args.positive_only,
                slices_per_input=args.slices_per_input,
                input_idx=args.input_idx,
                output_idx=args.output_idx,
                slice_axis=3,
                resize=args.resize)
    else:
        training_generator = sugen.DataGenerator(data_list=data_train_list,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                verbose=args.verbose, 
                residual_mode=args.residual_mode,
                positive_only = args.positive_only,
                slices_per_input=args.slices_per_input,
                input_idx=args.input_idx,
                output_idx=args.output_idx,
                slice_axis=args.slice_axis,
                resize=args.resize)

    if r > 0:
        if args.train_mpr:
            validation_generator_0 = sugen.DataGenerator(data_list=data_val_list,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    verbose=args.verbose, 
                    residual_mode=args.residual_mode,
                    positive_only = args.positive_only,
                    slices_per_input=args.slices_per_input,
                    input_idx=args.input_idx,
                    output_idx=args.output_idx,
                    slice_axis=0,
                    resize=args.resize)
            validation_generator_2 = sugen.DataGenerator(data_list=data_val_list,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    verbose=args.verbose, 
                    residual_mode=args.residual_mode,
                    positive_only = args.positive_only,
                    slices_per_input=args.slices_per_input,
                    input_idx=args.input_idx,
                    output_idx=args.output_idx,
                    slice_axis=2,
                    resize=args.resize)
            validation_generator_3 = sugen.DataGenerator(data_list=data_val_list,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    verbose=args.verbose, 
                    residual_mode=args.residual_mode,
                    positive_only = args.positive_only,
                    slices_per_input=args.slices_per_input,
                    input_idx=args.input_idx,
                    output_idx=args.output_idx,
                    slice_axis=3,
                    resize=args.resize)
        else:
            validation_generator = sugen.DataGenerator(data_list=data_val_list,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    verbose=args.verbose, 
                    residual_mode=args.residual_mode,
                    positive_only = args.positive_only,
                    slices_per_input=args.slices_per_input,
                    input_idx=args.input_idx,
                    output_idx=args.output_idx,
                    slice_axis=args.slice_axis,
                    resize=args.resize)
    else:
        validation_generator = None

    if args.train_mpr:
        for i in range(0, args.num_epochs, 3):
            print('##### EPOCH {} OF {} #####'.format(i+1, args.num_epochs))
            history = m.model.fit_generator(generator=training_generator_0, validation_data=validation_generator_0, validation_steps=args.val_steps_per_epoch, use_multiprocessing=args.use_multiprocessing, workers=args.num_workers, max_queue_size=args.max_queue_size, epochs=i+1, steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose, initial_epoch=i)
            history = m.model.fit_generator(generator=training_generator_2, validation_data=validation_generator_2, validation_steps=args.val_steps_per_epoch, use_multiprocessing=args.use_multiprocessing, workers=args.num_workers, max_queue_size=args.max_queue_size, epochs=i+2, steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose, initial_epoch=i+1)
            history = m.model.fit_generator(generator=training_generator_3, validation_data=validation_generator_3, validation_steps=args.val_steps_per_epoch, use_multiprocessing=args.use_multiprocessing, workers=args.num_workers, max_queue_size=args.max_queue_size, epochs=i+3, steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose, initial_epoch=i+2)
    else:
        history = m.model.fit_generator(generator=training_generator, validation_data=validation_generator, validation_steps=args.val_steps_per_epoch, use_multiprocessing=args.use_multiprocessing, workers=args.num_workers, max_queue_size=args.max_queue_size, epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose, initial_epoch=0)

    toc = time.time()
    print('done training ({:.0f} sec)'.format(toc - tic))

    if args.history_file is not None:
        np.save(args.history_file, history.history)
