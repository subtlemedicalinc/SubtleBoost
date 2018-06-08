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

usage_str = 'usage: %(prog)s [options]'
description_str = 'train SubtleGrad network on pre-processed data'

# FIXME: add time stamps, logging
# FIXME: data augmentation

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', action='store', dest='data_dir', type=str, help='directory containing pre-processed npy files', default=None)
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')
    parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs to run', default=10)
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--gpu', action='store', dest='gpu_device', type=str, help='set GPU', default=None)
    parser.add_argument('--keras_memory', action='store', dest='keras_memory', type=float, help='set Keras memory (0 to 1)', default=.8)
    parser.add_argument('--checkpoint', action='store', dest='checkpoint_file', type=str, help='checkpoint file', default=None)
    parser.add_argument('--validation_split', action='store', dest='val_split', type=float, help='ratio of validation data', default=.1)
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='RNG seed', default=723)
    parser.add_argument('--log_dir', action='store', dest='log_dir', type=str, help='log directory', default='logs')
    parser.add_argument('--max_data_sets', action='store', dest='max_data_sets', type=int, help='limit number of data sets', default=None)
    parser.add_argument('--predict', action='store', dest='predict_file', type=str, help='perform prediction and write to file', default=None)
    parser.add_argument('--learn_residual', action='store_true', dest='residual_mode', help='learn residual, (zero, low - zero, full - zero)', default=False)
    parser.add_argument('--learning_rate', action='store', dest='lr_init', type=float, help='intial learning rate', default=.001)
    parser.add_argument('--batch_norm', action='store_true', dest='batch_norm', help='batch normalization')


    args = parser.parse_args()
    
    verbose = args.verbose
    data_dir = args.data_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu_device = args.gpu_device
    keras_memory = args.keras_memory
    checkpoint_file = args.checkpoint_file
    val_split = args.val_split
    random_seed = args.random_seed
    log_dir = args.log_dir
    predict_file = args.predict_file
    residual_mode = args.residual_mode
    lr_init = args.lr_init
    batch_norm = args.batch_norm

    if log_dir is not None:
        try:
            os.mkdir(log_dir)
        except Exception as e:
            warn(str(e))
            pass
            pass

    if args.max_data_sets is None:
        max_data_sets = np.inf
    else:
        max_data_sets = args.max_data_sets

    assert data_dir is not None, 'must specify data directory'

    if gpu_device is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

    random.seed(random_seed)
    np.random.seed(random_seed)

    # load data
    if verbose:
        print('loading data from {}'.format(data_dir))
        tic = time.time()

    # each element of the data_list contains 3 sets of 3D
    # volumes containing zero, low, and full contrast.
    # the number of slices may differ but the image dimensions
    # should be the same

    npy_list = suio.get_npy_files(data_dir, max_data_sets=max_data_sets)
    random.shuffle(npy_list)

    # load initial file to get dimensions
    data = suio.load_npy_file(npy_list[0])
    #FIXME: check that image sizes are the same
    _, nx, ny, _ = data.shape

    sugn.clear_keras_memory()
    sugn.set_keras_memory(keras_memory)

    m = sugn.DeepEncoderDecoder2D(
            num_channel_input=2, num_channel_output=1,
            img_rows=nx, img_cols=ny,
            num_channel_first=32,
            lr_init=lr_init,
            batch_norm=batch_norm,
            verbose=verbose, checkpoint_file=checkpoint_file, log_dir=log_dir)

    m.load_weights()

    if predict_file is not None:

        tic = time.time()
        print('predicting...')
        Y_prediction = m.model.predict(X, batch_size=batch_size, verbose=verbose)
        toc = time.time()
        print('done with predicting ({:.0f} sec)'.format(toc - tic))
        np.save(predict_file, Y_prediction)
    else:

        print('training...')

        if verbose:
            print('epoch\tfile')

        for epoch in range(num_epochs):

            for npy_file in npy_list:

                if verbose:
                    print('{}\t{}'.format(epoch, npy_file))

                # load single volume
                data = suio.load_npy_file(npy_file)
                _ridx = np.random.permutation(data.shape[0])

                X = data[_ridx,:,:,:2]
                Y = data[_ridx,:,:,-1][:,:,:,None]

                #X = np.concatenate([dl[:,:,:,:2] for dl in data_list], axis=0)
                #Y = np.concatenate([dl[:,:,:,-1] for dl in data_list], axis=0)[:,:,:,None]

                if verbose:
                    print('X, Y sizes = ', X.shape, Y.shape)

                if residual_mode:
                    if verbose:
                        print('residual mode. train on (zero, low - zero, full - zero)')
                    X[:,:,:,1] -= X[:,:,:,0]
                    Y -= X[:,:,:,0][:,:,:,None]


                cb_checkpoint = m.callback_checkpoint()
                cb_tensorboard = m.callback_tensorbaord()

                history = m.model.fit(X, Y, batch_size=batch_size, epochs=1, validation_split=val_split, callbacks=[cb_checkpoint, cb_tensorboard], verbose=verbose)
                toc = time.time()
        print('done training ({:.0f} sec)'.format(toc - tic))
