import time
import itertools
import numpy as np

import sigpy as sp
import keras
from expiringdict import ExpiringDict

from subtle.subtle_io import build_block_list, load_file, load_blocks, is_valid_block

class BlockLoader(keras.utils.Sequence):
    def __init__(
        self, data_list, batch_size=8, block_size=64, block_strides=16, shuffle=True, verbose=1, predict=False, brain_only=None, brain_only_mode=None, load_full_volume=False, predict_full=False
    ):
        self.data_list = data_list
        self.batch_size = batch_size
        self.block_size = block_size
        self.block_strides = block_strides
        self.shuffle = shuffle
        self.verbose = verbose
        self.predict = predict
        self.predict_full = predict_full
        self.brain_only = brain_only
        self.brain_only_mode = brain_only_mode
        self.h5_key = 'data_mask' if self.brain_only else 'data'
        self.load_full_volume = load_full_volume

        self.ims_cache = ExpiringDict(max_len=100, max_age_seconds=24*60*60)

        self._init_block_info()
        self.on_epoch_end()

    def _init_block_info(self):
        self.block_list_files, self.block_list_indices = build_block_list(self.data_list, self.block_size, self.block_strides, params={'h5_key': self.h5_key})

        self.num_blocks = len(self.block_list_indices)

    def __len__(self):
        return int(np.ceil(self.num_blocks / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_blocks)
        if self.shuffle == True:
            self.indexes = np.random.permutation(self.indexes)

    def _get_ims(self, fpath):
        ims = self.ims_cache.get(fpath)
        if ims is not None:
            return ims[0], ims[1]

        (ims, ims_mask) = load_file(fpath, file_type='h5', params={'h5_key': 'all'})
        ims = ims.transpose(1, 0, 2, 3)
        ims_mask = ims_mask.transpose(1, 0, 2, 3)

        self._cache_img(fpath, ims, ims_mask)
        return ims, ims_mask

    def _cache_img(self, fpath, ims, ims_mask=None):
        self.ims_cache[fpath] = (ims, ims_mask)

    def _group_fetch(self, file_list, block_list):
        group_dict = {}
        for fname, bidx in zip(file_list, block_list):
            group_dict.setdefault(fname, []).append(bidx)

        block_list = []
        block_mask_list = []

        for fpath, block_idxs in group_dict.items():
            ims, ims_mask = self._get_ims(fpath)

            blocks = load_blocks(ims, indices=block_idxs, block_size=self.block_size, strides=self.block_strides)
            block_list.extend(blocks)

            block_masks = load_blocks(ims_mask, indices=block_idxs, block_size=self.block_size, strides=self.block_strides)
            block_mask_list.extend(block_masks)

        return np.array(block_list), np.array(block_mask_list)

    def __getitem__(self, index, enforce_raw_data=False):
        file_list = self.block_list_files[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
        block_list = self.block_list_indices[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]

        self._current_file_list = file_list

        if self.predict_full:
            X = np.array([self._get_ims(fpath)[0] for fpath in file_list])
            X = X[:, :, :2, ...].transpose(0, 3, 4, 1, 2)
            return X, None, None

        X = []
        Y = []
        weights = []

        gfetch = self._group_fetch(file_list, block_list)
        block_list = gfetch[0]
        block_mask_list = gfetch[0]

        for (blocks, block_masks) in zip(block_list, block_mask_list):
            x_item = blocks[:2]
            X.append(x_item)

            weights.append(int(is_valid_block(block_masks[0], block_size=self.block_size)))
            y_item = np.array([blocks[2]])
            Y.append(y_item)

        X = np.array(X)
        X = X.transpose(0, 2, 3, 4, 1)

        Y = np.array(Y)
        Y = Y.transpose(0, 2, 3, 4, 1)

        weights = np.array(weights)
        return X, Y, weights
