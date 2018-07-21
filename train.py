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
    parser.add_argument('--steps_per_epoch', action='store', dest='steps_per_epoch', type=int, help='number of iterations per epoch (default -- # slices in dataset / batch_size', default=None)
    parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, help='number of workers for generator', default=1)
    parser.add_argument('--shuffle', action='store_true', dest='shuffle', help='shuffle input data files each epoch', default=False)
    parser.add_argument('--history_file', action='store', dest='history_file', type=str, help='store history in npy file', default=None)


    args = parser.parse_args()
    
    assert args.num_workers == 1, "FIXME"


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

    assert args.data_dir is not None, 'must specify data directory'

    if args.gpu_device is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # load data
    if args.verbose:
        print('loading data from {}'.format(args.data_dir))
        tic = time.time()

    # each element of the data_list contains 3 sets of 3D
    # volumes containing zero, low, and full contrast.
    # the number of slices may differ but the image dimensions
    # should be the same

    npy_list = suio.get_npy_files(args.data_dir, max_data_sets=max_data_sets)
    random.shuffle(npy_list)

    # load initial file to get dimensions
    data = suio.load_npy_file(npy_list[0])
    #FIXME: check that image sizes are the same
    _, nx, ny, _ = data.shape

    sugn.clear_keras_memory()
    sugn.set_keras_memory(args.keras_memory)

    m = sugn.DeepEncoderDecoder2D(
            num_channel_input=2, num_channel_output=1,
            img_rows=nx, img_cols=ny,
            num_channel_first=32,
            lr_init=args.lr_init,
            batch_norm=args.batch_norm,
            verbose=args.verbose, checkpoint_file=args.checkpoint_file, log_dir=args.log_dir)

    m.load_weights()

    tic = time.time()
    if args.predict_dir is not None:

        print('predicting...')

        for npy_file in npy_list:

            if args.verbose:
                print('{}:'.format(npy_file))

            # load single volume
            data = suio.load_npy_file(npy_file)


            X = data[:,:,:,:2]
            #Y = data[:,:,:,-1][:,:,:,None]

            if args.verbose:
                print('X size = ', X.shape)

            if args.residual_mode:
                if args.verbose:
                    print('residual mode. train on (zero, low - zero, full - zero)')
                X[:,:,:,1] -= X[:,:,:,0]
                #X[:,:,:,1] = np.maximum(0., X[:,:,:,1])

            Y_prediction = X[:,:,:,0][:,:,:,None] + m.model.predict(X, batch_size=args.batch_size, verbose=args.verbose)

            npy_base = os.path.basename(npy_file)
            npy_file_predict = '{}/{}_predict.npy'.format(args.predict_dir, os.path.splitext(npy_base)[0])

            if args.verbose:
                print('output: {}'.format(npy_file_predict))

            np.save(npy_file_predict, Y_prediction)

        toc = time.time()
        print('done predicting ({:.0f} sec)'.format(toc - tic))

    else:

        print('training...')

        if args.val_split == 0. and len(npy_list) > 1:
            npy_list_val = npy_list[0]
            if args.verbose:
                print('using {} for validation'.format(npy_list_val))

            npy_list = npy_list[1:]

            data_val = suio.load_npy_file(npy_list_val)

            X_val = data_val[:,:,:,:2]
            Y_val = data_val[:,:,:,-1][:,:,:,None]

            if args.residual_mode:
                if args.verbose:
                    print('residual mode. train on (zero, low - zero, full - zero)')
                X_val[:,:,:,1] -= X_val[:,:,:,0]
                #X_val[:,:,:,1] = np.maximum(0., X_val[:,:,:,1])
                Y_val -= X_val[:,:,:,0][:,:,:,None]
                #Y_val = np.maximum(0., Y_val)


        cb_checkpoint = m.callback_checkpoint()
        #cb_tensorboard = m.callback_tensorbaord()
        cb_tensorboard = m.callback_tensorbaord(log_every=1)


        training_generator = suio.DataGenerator(npy_list=npy_list,
                batch_size=args.batch_size,
                num_channel_input=2, num_channel_output=1,
                img_rows=nx, img_cols=ny,
                shuffle=args.shuffle,
                verbose=args.verbose, 
                residual_mode=args.residual_mode)

        history = m.model.fit_generator(generator=training_generator, validation_data=(X_val, Y_val), use_multiprocessing=True, workers=args.num_workers, epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch, callbacks=[cb_checkpoint, cb_tensorboard], verbose=args.verbose)

        toc = time.time()
        print('done training ({:.0f} sec)'.format(toc - tic))

        if args.history_file is not None:
            np.save(args.history_file, history.history)
