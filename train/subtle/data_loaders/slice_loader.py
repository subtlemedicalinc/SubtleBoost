'''
2.5 generator for contrast synthesis

@author: Srivathsa Pasumarthi (srivathsa@subtlemedical.com)

Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2023/02/17
'''

import time

import os
import numpy as np
import sigpy as sp
from expiringdict import ExpiringDict
from tqdm import tqdm
from glob import glob
import random
from torch.utils.data import Dataset

from subtle.utils.slice import build_slice_list, get_num_slices
from subtle.subtle_preprocess import enh_mask_smooth

def get_context_slices(params):
    case_id = params['fpath'].split('/')[-3]
    ax_str = params['fpath'].split('/')[-2]
    min_idx, max_idx = params['min_max_range'][case_id][ax_str]

    fnum = int(params['fpath'].split('/')[-1].replace('.{}'.format(params['file_ext']), ''))
    delta = params['slices_per_input'] // 2

    sl_idxs = np.arange(fnum - delta, fnum + delta + 1)
    sl_idxs = np.clip(sl_idxs, min_idx, max_idx)

    num_mod = len(params['input_idx']) + len(params['output_idx']) + 1
    # + 1 for enh_mask that is pre_computed and stored in the data

    slices = np.zeros((params['slices_per_input'], num_mod, params['resize'], params['resize']))

    slice_dir = os.path.join(params['data_src'], case_id, ax_str)
    for idx, sl_idx in enumerate(sl_idxs):
        fp = os.path.join(slice_dir, '{:03d}.{}'.format(sl_idx, params['file_ext']))
        sl_data = np.load(fp)
        try:
            slices[idx, :] = sl_data
        except Exception as exc:
            print('ERROR in {}'.format(fp))

    return slices


class SliceLoader(Dataset):
    def __init__(
        self, data_files, slices_per_input=1, shuffle=True, verbose=1, predict=False, input_idx=[0, 1], output_idx=[2], slice_axis=[0], num_channel_output=1, use_enh_mask=False, enh_pfactor=1, file_ext='npy', resize=512, **args
    ):

        'Initialization'
        self.data_src = '/'.join(data_files[0].split('/')[:-1])
        self.case_ids = sorted([f.split('/')[-1] for f in data_files])

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
            sl_ax_list = self.slice_axis

            # randomize saggital and coronal views for each case. all cases have axial
            # if random.uniform(0, 1) > 0.5:
            #     sl_ax_list.extend(np.random.choice([2, 3], size=1))
            #     if random.uniform(0, 1) > 0.8:
            #         rem = [s for s in self.slice_axis if s not in sl_ax_list]
            #         sl_ax_list.extend(rem)

            for sl_ax in sl_ax_list:
                ax_str = mpr_map[sl_ax]
                src_dir = os.path.join(self.data_src, cnum, ax_str)
                slice_files = sorted([fp for fp in glob('{}/*.{}'.format(src_dir, self.file_ext))])
                all_slices.extend(slice_files)
                self.min_max_range[cnum][ax_str] = self._get_min_max_range(cnum, ax_str)

        self.slice_list_files = np.array(all_slices)
        self.num_slices = len(self.slice_list_files)

    def _get_min_max_range(self, case_id, slice_axis):
        src_dir = os.path.join(self.data_src, case_id, slice_axis)
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
        return self.num_slices

    def _fetch_slices(self, fpath):
        slices = self.ims_cache.get(fpath)

        if slices is None:
            slices = np.load(fpath)
            self.ims_cache[fpath] = slices
        return slices

    def __getitem__(self, index=0, fpath=None):
        t1 = time.time()
        fpath = fpath if fpath is not None else self.slice_list_files[index]

        proc_params = {
            'fpath': fpath,
            'min_max_range': self.min_max_range,
            'file_ext': self.file_ext,
            'slices_per_input': self.slices_per_input,
            'resize': self.resize,
            'input_idx': self.input_idx,
            'output_idx': self.output_idx,
            'data_src': self.data_src
        }

        data = get_context_slices(proc_params)
        data = data[None].astype(np.float32)

        t2 = time.time()

        X = data[:, :, self.input_idx, ...]
        X = X.reshape((X.shape[0], -1, X.shape[3], X.shape[4]), order='F')

        if self.predict:
            return X[0]

        h = self.slices_per_input // 2
        Y_data = data[:, h, self.output_idx, :].squeeze()
        enh_mask = np.zeros((1, self.resize, self.resize))
        if self.use_enh_mask:
            enh_mask = data[:, h, -1] ** self.enh_pfactor
            enh_mask = np.clip(enh_mask, 1e-3, data[:, h, -2].max())

        Y = np.zeros((1, 2, self.resize, self.resize))
        Y[:, 0] = Y_data
        Y[:, 1] = enh_mask

        t3 = time.time()

        # print('Part 1=', t2-t1, 'Part2=', t3-t2, 'Total=', t3-t1)
        return X[0], Y[0]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_slices)
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
