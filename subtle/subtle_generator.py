'''
subtle_io.py

Generators for contrast synthesis

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/08/24
'''

import sys
import os # FIXME: transition from os to pathlib
import pathlib
from warnings import warn
import time

import h5py

import numpy as np

import sigpy as sp

try:
    import pydicom
except:
    import dicom as pydicom
try:
    import keras
except:
    pass

from subtle.subtle_io import *
from subtle.subtle_preprocess import resample_slices


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_list, batch_size=8, slices_per_input=1, shuffle=True, verbose=1, residual_mode=True, positive_only=False, predict=False, input_idx=[0,1], output_idx=[2], resize=None, slice_axis=0, resample_size=None):

        'Initialization'
        self.data_list = data_list
        self.batch_size = batch_size
        self.slices_per_input = slices_per_input # 2.5d
        self.shuffle = shuffle
        self.verbose = verbose
        self.residual_mode = residual_mode
        self.predict = predict
        self.positive_only = positive_only
        self.slice_axis = slice_axis
        self.resize = resize
        self.resample_size = resample_size

        _slice_list_files, _slice_list_indexes = build_slice_list(self.data_list, slice_axis=self.slice_axis)
        self.slice_list_files = np.array(_slice_list_files)
        self.slice_list_indexes = np.array(_slice_list_indexes)

        self.slices_per_file_dict = {data_file: get_num_slices(data_file, axis=self.slice_axis) for data_file in self.data_list}
        self.num_slices = len(self.slice_list_files)

        self.input_idx = input_idx
        self.output_idx = output_idx

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_slices / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        if self.verbose > 1:
            print('batch index:', index)

        file_list = self.slice_list_files[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
        slice_list = self.slice_list_indexes[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]

        if self.verbose > 1:
            print('list of files and slices:')
            print(file_list)
            print(slice_list)

        # Generate data
        if self.predict:
            X = self.__data_generation(file_list, slice_list)
            return X
        else:
            tic = time.time()
            X, Y = self.__data_generation(file_list, slice_list)
            if self.verbose > 1:
                print('generated batch in {} s'.format(time.time() - tic))
            return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_slices)
        if self.shuffle == True:
            self.indexes = np.random.permutation(self.indexes)

    def __data_generation(self, slice_list_files, slice_list_indexes):
        'Generates data containing batch_size samples' 

        # load volumes
        # each element of the data_list contains 3 sets of 3D
        # volumes containing zero, low, and full contrast.
        # the number of slices may differ but the image dimensions
        # should be the same

        data_list_X = []
        if not self.predict:
            data_list_Y = []

        for f, c in zip(slice_list_files, slice_list_indexes):
            num_slices = self.slices_per_file_dict[f]
            h = self.slices_per_input // 2

            # 2.5d
            idxs = np.arange(c - h, c + h + 1)

            # handle edge cases for 2.5d by just repeating the boundary slices
            idxs = np.minimum(np.maximum(idxs, 0), num_slices - 1)

            # FIXME: don't train on slices with very little brain signal. Can remove them by checking mask size relative to image size

            tic = time.time()
            slices = load_slices(input_file=f, slices=idxs, dim=self.slice_axis) # [c, 3, ny, nz]

            if self.slice_axis == 0:
                pass
            if self.slice_axis == 1:
                assert False, 'invalid slice axis!, {}'.format(self.slice_axis)
            elif self.slice_axis == 2:
                slices = np.transpose(slices, (2, 1, 0, 3))
            elif self.slice_axis == 3:
                slices = np.transpose(slices, (3, 1, 0, 2))

            # if self.resize is not None and (self.resize > slices.shape[-1] or self.resize > slices.shape[-2]):
            if self.resize is not None:
                slices = sp.util.resize(slices, [slices.shape[0], slices.shape[1], self.resize, self.resize])

            if self.resample_size is not None:
                if self.verbose > 1:
                    print('Resampling slices to matrix size', self.resample_size)
                slices = resample_slices(slices, self.resample_size)


            if self.verbose > 1:
                print('loaded slices from {} in {} s'.format(f, time.time() - tic))

            slices_X = slices[:,self.input_idx,:,:][None,...]
            data_list_X.append(slices_X)

            if not self.predict:
                slices_Y = slices[h, self.output_idx, :, :][None,None,...]
                data_list_Y.append(slices_Y)
            
        tic = time.time()
        if len(data_list_X) > 1:
            data_X = np.concatenate(data_list_X, axis=0)
            if not self.predict:
                data_Y = np.concatenate(data_list_Y, axis=0)
        else:
            data_X = data_list_X[0]
            if not self.predict:
                data_Y = data_list_Y[0]

        X = data_X.copy()
        if not self.predict:
            Y = data_Y.copy()
        if self.verbose > 1:
            print('reshaped data in {} s'.format(time.time() - tic))


        tic = time.time()
        if self.residual_mode and len(self.input_idx) == 2 and len(self.output_idx) == 1:
            if self.verbose > 1:
                print('residual mode. train on (zero, low - zero, full - zero)')
            X[:,:,1,:,:] -= X[:,:,0,:,:]
            if self.positive_only:
                X[:,:,1,:,:] = np.maximum(0, X[:,:,1,:,:])
            if not self.predict:
                Y -= X[:,h,0,:,:]
                if self.positive_only:
                    Y = np.maximum(0, Y)
        if self.verbose > 1:
            print('reisdual mode in {} s'.format(time.time() - tic))

        tic = time.time()
        # dims are [batch, slices_per_input, len(input_idx), nx, ny]
        # reshape to [batch, -1, nx, ny]
        # then transpose to [batch, nx, ny, -1]
        X = np.transpose(np.reshape(X, (X.shape[0], -1, X.shape[3], X.shape[4])), (0, 2, 3, 1))
        if not self.predict:
            Y = np.transpose(np.reshape(Y, (Y.shape[0], -1, Y.shape[3], Y.shape[4])), (0, 2, 3, 1))

        if self.verbose > 1:
            print('tranpose data in {} s'.format(time.time() - tic))

        if self.verbose > 1:
            if self.predict:
                print('X, size = ', X.shape)
            else:
                print('X, Y sizes = ', X.shape, Y.shape)


        if self.predict:
            return X
        else:
            return X, Y


