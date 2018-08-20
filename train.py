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
import argparse

import numpy as np

import keras.callbacks

import subtle.subtle_gad_network as sugn
import subtle.subtle_io as suio
import subtle.subtle_preprocess as sup


usage_str = 'usage: %(prog)s [options]'
description_str = 'train SubtleGrad network on pre-processed data'

# FIXME: add time stamps, logging
# FIXME: data augmentation


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_list', action='store', dest='data_list_file', type=str, help='list of pre-processed files for training', default=None)
    parser.add_argument('--data_dir', action='store', dest='data_dir', type=str, help='location of data', default=None)
    parser.add_argument('--file_ext', action='store', dest='file_ext', type=str, help='file extension of data', default=None)
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')
    parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs to run', default=10)
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, help='batch size', default=8)
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
    parser.add_argument('--use_multiprocessing', action='store_true', dest='use_multiprocessing', help='use multiprocessing in generator', default=False)
    parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, help='number of workers for generator', default=1)
    parser.add_argument('--max_queue_size', action='store', dest='max_queue_size', type=int, help='generator queue size', default=16)
    parser.add_argument('--shuffle', action='store_true', dest='shuffle', help='shuffle input data files each epoch', default=False)
    parser.add_argument('--history_file', action='store', dest='history_file', type=str, help='store history in npy file', default=None)
    parser.add_argument('--id', action='store', dest='job_id', type=str, help='job id for logging', default='')
    parser.add_argument('--slices_per_input', action='store', dest='slices_per_input', type=int, help='number of slices per input (2.5D)', default=1)
    parser.add_argument('--predict_file_ext', action='store', dest='predict_file_ext', type=str, help='file extension of predcited data', default='npy')


    args = parser.parse_args()
    
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

    # load data
    if args.verbose:
        print('loading data from {}'.format(args.data_list_file))
    tic = time.time()

    f = open(args.data_list_file, 'r')
    data_list = []
    for l in f.readlines():
        s = l.strip()
        if args.data_dir is not None:
            s = '{}/{}'.format(args.data_dir, s)
        if args.file_ext is not None:
            s = '{}.{}'.format(s, args.file_ext)
        data_list.append(s)
    f.close()

    # each element of the data_list contains 3 sets of 3D
    # volumes containing zero, low, and full contrast.
    # the number of slices may differ but the image dimensions
    # should be the same

    _ridx = np.random.permutation(len(data_list))
    data_list = [data_list[i] for i in _ridx[:args.max_data_sets]]

    # get dimensions from first file
    data_shape = suio.get_shape(data_list[0])
    #FIXME: check that image sizes are the same
    _, _, nx, ny = data_shape

    sugn.clear_keras_memory()
    sugn.set_keras_memory(args.keras_memory)

    m = sugn.DeepEncoderDecoder2D(
            num_channel_input=2 * args.slices_per_input, num_channel_output=1,
            img_rows=nx, img_cols=ny,
            num_channel_first=32,
            lr_init=args.lr_init,
            batch_norm=args.batch_norm,
            verbose=args.verbose, checkpoint_file=args.checkpoint_file, log_dir=args.log_dir, job_id=args.job_id)

    m.load_weights()

    tic = time.time()
    if args.predict_dir is not None:

        print('predicting...')

        for data_file in data_list:

            if args.verbose:
                print('{}:'.format(data_file))

            # use generator to maintain consistent data formatting
            prediction_generator = suio.DataGenerator(data_list=[data_file],
                    batch_size=args.batch_size,
                    shuffle=False,
                    verbose=args.verbose, 
                    residual_mode=args.residual_mode,
                    slices_per_input=args.slices_per_input)

            Y_prediction = m.model.predict_generator(generator=prediction_generator, max_queue_size=args.max_queue_size, workers=args.num_workers, use_multiprocessing=args.use_multiprocessing, verbose=args.verbose)

            # if residual mode is on, we need to load the data again
            # so that we can add the original contrast back in
            if args.residual_mode:
                data = suio.load_file(data_file).transpose((0, 2, 3, 1))
                h = args.slices_per_input // 2
                Y_prediction = data[:,:,:,0].squeeze() + Y_prediction.squeeze()

            data_file_base = os.path.basename(data_file)
            _1, _2 = os.path.splitext(data_file_base)
            data_file_predict = '{}/{}_predict_{}.{}'.format(args.predict_dir, _1, args.job_id, args.predict_file_ext)

            if args.verbose:
                print('output: {}'.format(data_file_predict))

            suio.save_data(data_file_predict, Y_prediction, file_type=args.predict_file_ext)

        toc = time.time()
        print('done predicting ({:.0f} sec)'.format(toc - tic))

    else:

        print('training...')

        if len(data_list) == 1:
            r = 0
        elif args.validation_split == 0:
            r = 1
        else: # len(data_list) > 1
            r = int(len(data_list) * args.validation_split)

        if args.verbose:
            print('using {} datasets for validation'.format(r))

        data_val_list = data_list[:r]
        data_train_list = data_list[r:]

        callbacks = []
        callbacks.append(m.callback_checkpoint())
        callbacks.append(m.callback_tensorbaord())
        callbacks.append(m.callback_tbimage(data_list=data_val_list, slice_dict_list=None, slices_per_epoch=1, slices_per_input=args.slices_per_input, batch_size=args.batch_size, verbose=args.verbose, residual_mode=args.residual_mode, tag='Image Example'))
        #cb_tensorboard = m.callback_tensorbaord(log_every=1)


        training_generator = suio.DataGenerator(data_list=data_train_list,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                verbose=args.verbose, 
                residual_mode=args.residual_mode,
                slices_per_input=args.slices_per_input)

        if r > 0:
            validation_generator = suio.DataGenerator(data_list=data_val_list,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    verbose=args.verbose, 
                    residual_mode=args.residual_mode,
                    slices_per_input=args.slices_per_input)
        else:
            validation_generator = None

        history = m.model.fit_generator(generator=training_generator, validation_data=validation_generator, validation_steps=8, use_multiprocessing=args.use_multiprocessing, workers=args.num_workers, max_queue_size=args.max_queue_size, epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose)

        toc = time.time()
        print('done training ({:.0f} sec)'.format(toc - tic))

        if args.history_file is not None:
            np.save(args.history_file, history.history)
