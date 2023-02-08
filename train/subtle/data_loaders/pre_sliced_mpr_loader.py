'''
Generators for contrast synthesis

@author: Srivathsa Pasumarthi (srivathsa@subtlemedical.com)

Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2022/12/17
'''

import time

import os
import numpy as np
import sigpy as sp
import keras
from expiringdict import ExpiringDict
from tqdm import tqdm
from glob import glob
import random

from subtle.utils.slice import build_slice_list, get_num_slices
from subtle.subtle_preprocess import enh_mask_smooth

class PreSlicedMPRLoader(keras.utils.Sequence):
    def __init__(
        self, data_list, batch_size=8, slices_per_input=1, shuffle=True, verbose=1, predict=False, input_idx=[0, 1], output_idx=[2], slice_axis=[0], num_channel_output=1, use_enh_mask=False, enh_pfactor=1, file_ext='npy', resize=512, **args
    ):

        'Initialization'
        self.data_list = data_list
        self.case_ids = sorted([
            fp.split('/')[-1].replace('.{}'.format(file_ext), '')
            for fp in self.data_list
        ])
        self.data_src_path = '/'.join(self.data_list[0].split('/')[:-1])

        self.batch_size = batch_size
        self.slices_per_input = slices_per_input # 2.5d
        self.shuffle = shuffle
        self.verbose = verbose
        self.slice_axis = slice_axis
        self.h5_key = 'data'

        self.use_enh_mask = use_enh_mask
        self.enh_pfactor = enh_pfactor
        self.file_ext = file_ext
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.resize = resize
        self.predict = predict

        self.ims_cache = ExpiringDict(max_len=float("inf"), max_age_seconds=3*24*60*60)

        '''
        The below dict keeps track of the min and max indices of each case in each orientation.
        This is required to clip the indices at the slice boundaries
        '''
        self.min_max_range = {}
        self._init_slice_info()
        # self._init_img_cache()
        self.on_epoch_end()

    def _init_slice_info(self):
        mpr_map = {0: 'ax', 2: 'sag', 3: 'cor'}

        all_slices = []

        for cnum in self.case_ids:
            self.min_max_range[cnum] = {}
            sl_ax_list = [0]

            # randomize saggital and coronal views for each case. all cases have axial
            if random.uniform(0, 1) > 0.5:
                sl_ax_list.extend(np.random.choice([2, 3], size=1))
                if random.uniform(0, 1) > 0.8:
                    rem = [s for s in self.slice_axis if s not in sl_ax_list]
                    sl_ax_list.extend(rem)

            for sl_ax in sl_ax_list:
                ax_str = mpr_map[sl_ax]
                src_dir = os.path.join(self.data_src_path, cnum, ax_str)
                slice_files = sorted([fp for fp in glob('{}/*.{}'.format(src_dir, self.file_ext))])
                all_slices.extend(slice_files)
                self.min_max_range[cnum][ax_str] = self._get_min_max_range(cnum, ax_str)

        self.slice_list_files = np.array(all_slices)
        self.num_slices = len(self.slice_list_files)

    def _get_min_max_range(self, case_id, slice_axis):
        src_dir = os.path.join(self.data_src_path, case_id, slice_axis)
        fps = sorted([f for f in glob('{}/*.{}'.format(src_dir, self.file_ext))])
        get_file_idx = lambda fp: int(fp.split('/')[-1].replace('.{}'.format(self.file_ext), ''))
        min_idx = get_file_idx(fps[0])
        max_idx = get_file_idx(fps[-1])
        return (min_idx, max_idx)

    def _init_img_cache(self):
        print('Initializing image cache...')
        for fpath in tqdm(self.slice_list_files, total=len(self.slice_list_files)):
            _ = self._fetch_slices(fpath)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_slices / self.batch_size))

    def _fetch_slices(self, fpath):
        slices = self.ims_cache.get(fpath)

        if slices is None:
            slices = np.load(fpath)
            self.ims_cache[fpath] = slices
        return slices

    def _get_context_slices(self, fpath):
        case_id = fpath.split('/')[-3]
        ax_str = fpath.split('/')[-2]
        min_idx, max_idx = self.min_max_range[case_id][ax_str]

        fnum = int(fpath.split('/')[-1].replace('.{}'.format(self.file_ext), ''))
        delta = self.slices_per_input // 2

        sl_idxs = np.arange(fnum - delta, fnum + delta + 1)
        sl_idxs = np.clip(sl_idxs, min_idx, max_idx)

        num_mod = len(self.input_idx) + len(self.output_idx)
        slices = np.zeros((self.slices_per_input, num_mod, self.resize, self.resize))
        slices_mask = np.zeros((self.slices_per_input, num_mod, self.resize, self.resize))

        slice_dir = os.path.join(self.data_src_path, case_id, ax_str)
        for idx, sl_idx in enumerate(sl_idxs):
            fp = os.path.join(slice_dir, '{:03d}.{}'.format(sl_idx, self.file_ext))
            data, data_mask = self._fetch_slices(fp)

            slices[idx, :] = data
            slices_mask[idx, :] = data_mask

        return slices, slices_mask

    def __getitem__(self, index):
        t1 = time.time()

        file_list = self.slice_list_files[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]

        num_mod = len(self.input_idx) + len(self.output_idx)
        data = np.zeros(
            (self.batch_size, self.slices_per_input, num_mod, self.resize, self.resize)
        )
        data_mask = np.zeros(
            (self.batch_size, self.slices_per_input, num_mod, self.resize, self.resize)
        )

        for idx, fpath in enumerate(file_list):
            slices, slices_mask = self._get_context_slices(fpath)
            data[idx, :] = slices
            data_mask[idx, :] = slices_mask

        t2 = time.time()

        X = data[:, :, self.input_idx, ...]
        X = X.reshape((X.shape[0], -1, X.shape[3], X.shape[4]), order='F')
        X = X.transpose(0, 2, 3, 1)

        if self.predict:
            return X

        h = self.slices_per_input // 2
        Y_data = data[:, h, self.output_idx, :].squeeze()

        enh_mask = np.zeros((self.batch_size, self.resize, self.resize))
        if self.use_enh_mask:
            X_input = data_mask[:, :, self.input_idx, ...]
            Y_input = data_mask[:, [h], self.output_idx, ...][:, :, None, ...]
            enh_mask = enh_mask_smooth(X_input, Y_input, center_slice=h, p=self.enh_pfactor).squeeze()

        Y = np.zeros((self.batch_size, self.resize, self.resize, 2))
        Y[..., 0] = Y_data
        Y[..., 1] = enh_mask

        t3 = time.time()

        print('Part 1=', t2-t1, 'Part2=', t3-t2, 'Total=', t3-t1)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_slices)
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
