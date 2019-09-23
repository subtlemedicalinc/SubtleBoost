import numpy as np

import sigpy as sp
import keras

from subtle.subtle_io import build_block_list, load_blocks, is_valid_block

class BlockLoader(keras.utils.Sequence):
    def __init__(
        self, data_list, batch_size=8, block_size=64, strides=16, shuffle=True, verbose=1, predict=False, brain_only=None, brain_only_mode=None
    ):
        self.data_list = data_list
        self.batch_size = batch_size
        self.block_size = block_size
        self.strides = strides
        self.shuffle = shuffle
        self.verbose = verbose
        self.predict = predict
        self.brain_only = brain_only
        self.brain_only_mode = brain_only_mode
        self.h5_key = 'data_mask' if self.brain_only else 'data'

        self._init_block_info()
        self.on_epoch_end()

    def _init_block_info(self):
        self.block_list_files, self.block_list_indices = build_block_list(self.data_list, self.block_size, self.strides, params={'h5_key': self.h5_key})

        self.num_blocks = len(self.block_list_indices)

    def __len__(self):
        return int(np.floor(self.num_blocks / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_blocks)
        if self.shuffle == True:
            self.indexes = np.random.permutation(self.indexes)

    def _group_fetch(self, file_list, block_list):
        group_dict = {}
        idx_list = []
        for fpath, idx in zip(file_list, block_list):
            if fpath in group_dict:
                idx_list.append(len(group_dict[fpath]))
                group_dict[fpath].append(idx)
            else:
                idx_list.append(0)
                group_dict[fpath] = [idx]

        block_list = [None] * len(file_list)

        for fpath, block_idxs in group_dict.items():
            blocks = load_blocks(fpath, indices=block_idxs, block_size=self.block_size, strides=self.strides, params={'h5_key': self.h5_key})

            idxs = [i for i, v in enumerate(file_list) if v == fpath]
            for n, i in enumerate(idxs):
                block_list[i] = blocks[n]

        return np.array(block_list)


    def __getitem__(self, index):
        file_list = self.block_list_files[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
        block_list = self.block_list_indices[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]

        X = []
        Y = []
        weights = []
        for blocks in self._group_fetch(file_list, block_list):
            x_item = blocks[:2]
            X.append(x_item)

            if not self.predict:
                weights.append(is_valid_block(blocks[0], block_size=self.block_size))
                y_item = np.array(blocks[2])
                Y.append(y_item)

        if self.predict:
            return np.array(X)

        return np.array(X), np.array(Y), np.array(weights)
