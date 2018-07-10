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

def normalize_data(data, verbose=False):
    ntic = time.time()
    if verbose:
        print('normalizing data')
    data = sup.normalize_im(data, axis=(0,1,2)) # normalize each contrast separately
    ntoc = time.time()
    if verbose:
        print('done ({:.2f}s)'.format(ntoc - ntic))

    if verbose:
        print('scaling data')
    ntic = time.time()
    nz = data.shape[0]
    idx_scale = range(nz//2 - 5, nz//2 + 5)
    scale_low = sup.scale_im_enhao(data[idx_scale, :, :, 0], data[idx_scale, :, :, 1])
    scale_full = sup.scale_im_enhao(data[idx_scale, :, :, 0], data[idx_scale, :, :, 2])
    ntoc = time.time()
    if verbose:
        print('scale low:', scale_low)
        print('scale full:', scale_full)
        print('done scaling data ({:.2f} s)'.format(ntoc - ntic))
    data[:,:,:,1] /= scale_low
    data[:,:,:,2] /= scale_full

    return data

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
    parser.add_argument('--predict', action='store', dest='predict_dir', type=str, help='perform prediction and write to directory', default=None)
    parser.add_argument('--learn_residual', action='store_true', dest='residual_mode', help='learn residual, (zero, low - zero, full - zero)', default=False)
    parser.add_argument('--learning_rate', action='store', dest='lr_init', type=float, help='intial learning rate', default=.001)
    parser.add_argument('--batch_norm', action='store_true', dest='batch_norm', help='batch normalization')
    parser.add_argument('--dont_normalize', action='store_false', dest='normalize', help='turn off data normalization')
    parser.add_argument('--data_per_fit', action='store', dest='data_per_fit', type=int, help='number of data sets per call to fit', default=1)


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
    predict_dir = args.predict_dir
    residual_mode = args.residual_mode
    lr_init = args.lr_init
    batch_norm = args.batch_norm
    normalize = args.normalize
    data_per_fit = args.data_per_fit

    if log_dir is not None:
        try:
            os.mkdir(log_dir)
        except Exception as e:
            warn(str(e))
            pass

    if predict_dir is not None:
        try:
            os.mkdir(predict_dir)
        except Exception as e:
            warn(str(e))
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

    tic = time.time()
    if predict_dir is not None:

        print('predicting...')

        for npy_file in npy_list:

            if verbose:
                print('{}:'.format(npy_file))

            # load single volume
            data = suio.load_npy_file(npy_file)

            if normalize:
                data = normalize_data(data, verbose)

            X = data[:,:,:,:2]
            #Y = data[:,:,:,-1][:,:,:,None]

            if verbose:
                print('X size = ', X.shape)

            if residual_mode:
                if verbose:
                    print('residual mode. train on (zero, low - zero, full - zero)')
                X[:,:,:,1] -= X[:,:,:,0]
                #X[:,:,:,1] = np.maximum(0., X[:,:,:,1])

            Y_prediction = X[:,:,:,0][:,:,:,None] + m.model.predict(X, batch_size=batch_size, verbose=verbose)

            npy_base = os.path.basename(npy_file)
            npy_file_predict = '{}/{}_predict.npy'.format(predict_dir, os.path.splitext(npy_base)[0])

            if verbose:
                print('output: {}'.format(npy_file_predict))

            np.save(npy_file_predict, Y_prediction)

        toc = time.time()
        print('done predicting ({:.0f} sec)'.format(toc - tic))

    else:

        print('training...')

        if val_split == 0. and len(npy_list) > 1:
            npy_list_val = npy_list[0]
            if verbose:
                print('using {} for validation'.format(npy_list_val))

            npy_list = npy_list[1:]

            data_val = suio.load_npy_file(npy_list_val)
            if normalize:
                data_val = normalize_data(data_val, verbose)
                if verbose:
                    print('mean of data:', np.mean(data, axis=(0,1,2)))

            X_val = data_val[:,:,:,:2]
            Y_val = data_val[:,:,:,-1][:,:,:,None]

            if residual_mode:
                if verbose:
                    print('residual mode. train on (zero, low - zero, full - zero)')
                X_val[:,:,:,1] -= X_val[:,:,:,0]
                #X_val[:,:,:,1] = np.maximum(0., X_val[:,:,:,1])
                Y_val -= X_val[:,:,:,0][:,:,:,None]
                #Y_val = np.maximum(0., Y_val)



        if verbose:
            print('epoch\tfile')

        cb_checkpoint = m.callback_checkpoint()
        cb_tensorboard = m.callback_tensorbaord()


        for epoch in range(num_epochs):

            _npy_list = list(npy_list)

            i_npy = 0
            while len(_npy_list) > 0:

                i_npy += 1

                npy_files = _npy_list[:data_per_fit]
                del _npy_list[:data_per_fit]



                if verbose:
                    print('{0}/{1}\t{2}/{3}\t{4}'.format(epoch, num_epochs, i_npy, int(np.ceil(len(npy_list) / data_per_fit)), npy_files))

                # load volumes
                # each element of the data_list contains 3 sets of 3D
                # volumes containing zero, low, and full contrast.
                # the number of slices may differ but the image dimensions
                # should be the same
                data_list = suio.load_npy_files('', npy_files)

                if normalize:
                    data_list = [normalize_data(d, verbose) for d in data_list]
                    if verbose:
                        print('mean of data:', [np.mean(d, axis=(0,1,2)) for d in data_list])

                data = np.concatenate(data_list, axis=0)
                _ridx = np.random.permutation(data.shape[0])

                X = data[_ridx,:,:,:2]
                Y = data[_ridx,:,:,-1][:,:,:,None]

                if verbose:
                    print('X, Y sizes = ', X.shape, Y.shape)

                if residual_mode:
                    if verbose:
                        print('residual mode. train on (zero, low - zero, full - zero)')
                    X[:,:,:,1] -= X[:,:,:,0]
                    #X[:,:,:,1] = np.maximum(0., X[:,:,:,1])
                    Y -= X[:,:,:,0][:,:,:,None]
                    #Y = np.maximum(0., Y)


                if val_split > 0:
                    history = m.model.fit(X, Y, batch_size=batch_size, epochs=1, validation_split=val_split, callbacks=[cb_checkpoint, cb_tensorboard], verbose=verbose)
                else:
                    if i_npy % 10 == 0:
                        history = m.model.fit(X, Y, batch_size=batch_size, epochs=1, validation_data=(X_val, Y_val), callbacks=[cb_checkpoint, cb_tensorboard], verbose=verbose)
                    else:
                        history = m.model.fit(X, Y, batch_size=batch_size, epochs=1, callbacks=[cb_checkpoint, cb_tensorboard], verbose=verbose)
                toc = time.time()
        toc = time.time()
        print('done training ({:.0f} sec)'.format(toc - tic))
