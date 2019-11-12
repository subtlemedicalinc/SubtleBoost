'''
Generators for contrast synthesis

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/08/24
'''

import time

import numpy as np
import sigpy as sp
import keras

from subtle.utils.slice import build_slice_list, get_num_slices
from subtle.utils.io import load_slices
from subtle.subtle_preprocess import resample_slices, enh_mask_smooth

class SliceLoader(keras.utils.Sequence):
    def __init__(self, data_list, batch_size=8, slices_per_input=1, shuffle=True, verbose=1, residual_mode=False, positive_only=False, predict=False, input_idx=[0, 1], output_idx=[2], resize=None, slice_axis=0, resample_size=None, brain_only=None, brain_only_mode=None, use_enh_mask=False, enh_pfactor=1.0):

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
        self.brain_only = brain_only
        self.brain_only_mode = brain_only_mode
        self.h5_key = 'data_mask' if self.brain_only else 'data'
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.enh_pfactor = enh_pfactor
        self.use_enh_mask = use_enh_mask

        self._init_slice_info()
        self.on_epoch_end()

    def _init_slice_info(self):
        _slice_list_files, _slice_list_indexes = build_slice_list(self.data_list, slice_axis=self.slice_axis, params={'h5_key': self.h5_key})

        self.slice_list_files = np.array(_slice_list_files)
        self.slice_list_indexes = np.array(_slice_list_indexes)

        self.slices_per_file_dict = {
            data_file: [get_num_slices(
                data_file,
                axis=sl_axis,
                params={'h5_key': self.h5_key}
            ) for sl_axis in self.slice_axis]
            for data_file in self.data_list
        }

        self.num_slices = len(self.slice_list_files)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_slices / self.batch_size))

    def __getitem__(self, index, enforce_raw_data=False, data_npy=None):
        'Generate one batch of data'

        if self.verbose > 1:
            print('batch index:', index)

        file_list = self.slice_list_files[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
        slice_list = self.slice_list_indexes[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]

        self._current_file_list = file_list
        self._current_slice_list = slice_list

        if self.verbose > 1:
            print('list of files and slices:')
            print(file_list)
            print(slice_list)

        # Generate data
        if self.predict:
            X = self._data_generation(file_list, slice_list, enforce_raw_data, data_npy)
            return X
        else:
            tic = time.time()
            X, Y = self._data_generation(file_list, slice_list, enforce_raw_data, data_npy)

            if self.verbose > 1:
                print('generated batch in {} s'.format(time.time() - tic))

            return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_slices)
        if self.shuffle == True:
            self.indexes = np.random.permutation(self.indexes)

    def _data_generation(self, slice_list_files, slice_list_indexes, enforce_raw_data=False, data_npy=None):
        'Generates data containing batch_size samples'

        # load volumes
        # each element of the data_list contains 3 sets of 3D
        # volumes containing zero, low, and full contrast.
        # the number of slices may differ but the image dimensions
        # should be the same

        data_list_X = []
        data_list_X_mask = []
        if not self.predict:
            data_list_Y = []
            data_list_Y_mask = []

        for i, (f, idx_dict) in enumerate(zip(slice_list_files, slice_list_indexes)):
            c = idx_dict['index']
            ax = idx_dict['axis']
            ax_pos = self.slice_axis.index(ax)

            num_slices = [num for i, num in enumerate(self.slices_per_file_dict[f]) if i == ax_pos][0]
            h = self.slices_per_input // 2

            # 2.5d
            idxs = np.arange(c - h, c + h + 1)

            # handle edge cases for 2.5d by just repeating the boundary slices
            idxs = np.minimum(np.maximum(idxs, 0), num_slices - 1)
            tic = time.time()

            h5_key = 'data_mask' if self.brain_only else 'data'
            if enforce_raw_data or (self.brain_only_mode == 'mixed' and i >= 5):
                h5_key = 'data'

            if data_npy is not None:
                slices = data_npy[idxs]
                slices_mask = data_npy[idxs]
            else:
                # load both full and brain-masked slices. Dimensions of each np array are [c, 3, ny, nz]
                # FIXME: only load slices_mask if BET masking was run, and prediction is False
                slices, slices_mask = load_slices(input_file=f, slices=idxs, dim=ax, params={'h5_key': 'all'})

            if ax == 0:
                pass
            if ax == 1:
                assert False, 'invalid slice axis!, {}'.format(ax)
            elif ax == 2:
                slices = np.transpose(slices, (2, 1, 0, 3))
                slices_mask = np.transpose(slices_mask, (2, 1, 0, 3))
            elif ax == 3:
                slices = np.transpose(slices, (3, 1, 0, 2))
                slices_mask = np.transpose(slices_mask, (3, 1, 0, 2))

            if self.resize is not None:
                slices = sp.util.resize(slices, [slices.shape[0], slices.shape[1], self.resize, self.resize])
                slices_mask = sp.util.resize(slices_mask, [slices_mask.shape[0], slices_mask.shape[1], self.resize, self.resize])

            if self.resample_size is not None:
                if self.verbose > 1:
                    print('Resampling slices to matrix size', self.resample_size)
                slices = resample_slices(slices, self.resample_size)
                slices_mask = resample_slices(slices_mask, self.resample_size)


            if self.verbose > 1:
                print('loaded slices from {} in {} s'.format(f, time.time() - tic))

            slices_X = slices[:,self.input_idx,:,:][None,...]
            slices_X_mask = slices_mask[:,self.input_idx,:,:][None,...]
            data_list_X.append(slices_X)
            data_list_X_mask.append(slices_X_mask)

            if not self.predict:
                slices_Y = slices[h, self.output_idx, :, :][None,None,...]
                slices_Y_mask = slices_mask[h, self.output_idx, :, :][None,None,...]
                data_list_Y.append(slices_Y)
                data_list_Y_mask.append(slices_Y_mask)

        tic = time.time()
        if len(data_list_X) > 1:
            data_X = np.concatenate(data_list_X, axis=0)
            data_X_mask = np.concatenate(data_list_X_mask, axis=0)
            if not self.predict:
                data_Y = np.concatenate(data_list_Y, axis=0)
                data_Y_mask = np.concatenate(data_list_Y_mask, axis=0)
        else:
            data_X = data_list_X[0]
            data_X_mask = data_list_X_mask[0]
            if not self.predict:
                data_Y = data_list_Y[0]
                data_Y_mask = data_list_Y_mask[0]

        X = data_X.copy()
        X_mask = data_X_mask.copy()
        if not self.predict:
            Y = data_Y.copy()
            Y_mask = data_Y_mask.copy()
            if self.use_enh_mask:
                enh_mask = enh_mask_smooth(X_mask, Y_mask, center_slice=h, p=self.enh_pfactor)
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
            print('residual mode in {} s'.format(time.time() - tic))

        tic = time.time()
        # dims are [batch, slices_per_input, len(input_idx), nx, ny]
        # reshape to [batch, -1, nx, ny]
        # then transpose to [batch, nx, ny, -1]
        X = np.transpose(np.reshape(X, (X.shape[0], -1, X.shape[3], X.shape[4])), (0, 2, 3, 1))

        # interleave - tmp code begin
        # X = X.transpose(3, 0, 1, 2)
        # tmp = [None]*(X.shape[0])
        # tmp[::2] = X[:self.slices_per_input]
        # tmp[1::2] = X[self.slices_per_input:]
        # X = np.array(tmp).transpose(1, 2, 3, 0)
        # interleave - tmp code end

        X_mask = np.transpose(np.reshape(X_mask, (X_mask.shape[0], -1, X_mask.shape[3], X_mask.shape[4])), (0, 2, 3, 1))

        if not self.predict:
            Y = np.transpose(np.reshape(Y, (Y.shape[0], -1, Y.shape[3], Y.shape[4])), (0, 2, 3, 1))
            Y_mask = np.transpose(np.reshape(Y_mask, (Y_mask.shape[0], -1, Y_mask.shape[3], Y_mask.shape[4])), (0, 2, 3, 1))
            if self.use_enh_mask:
                enh_mask = np.transpose(np.reshape(enh_mask, (enh_mask.shape[0], -1, enh_mask.shape[3], enh_mask.shape[4])), (0, 2, 3, 1))
            else:
                enh_mask = np.ones(Y.shape)
            Y = np.concatenate((Y, enh_mask), axis=-1)

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
