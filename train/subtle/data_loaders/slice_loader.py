'''
Generators for contrast synthesis

@author: Jon Tamir (jon@subtlemedical.com)
@author: Srivathsa Pasumarthi (srivathsa@subtlemedical.com)

Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/08/24
'''

import time

import os
import numpy as np
import sigpy as sp
import keras
from expiringdict import ExpiringDict
from tqdm import tqdm

from subtle.utils.slice import build_slice_list, get_num_slices
from subtle.utils.io import load_slices, load_file, load_h5_metadata
from subtle.subtle_preprocess import resample_slices, enh_mask_smooth, get_enh_mask_t2

class SliceLoader(keras.utils.Sequence):
    def __init__(self, data_list, batch_size=8, slices_per_input=1, shuffle=True, verbose=1, residual_mode=False, positive_only=False, predict=False, input_idx=[0, 1], output_idx=[2], resize=None, slice_axis=0, resample_size=None, brain_only=None, brain_only_mode=None, use_enh_mask=False, enh_pfactor=1.0, file_ext='npy', use_enh_uad=False, use_uad_ch_input=False, uad_ip_channels=1, fpath_uad_masks=[], uad_mask_threshold=0.1, uad_mask_path=None, uad_file_ext=None, enh_mask_t2=False, num_channel_output=1, multi_slice_gt=False):
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

        self.use_enh_uad = use_enh_uad
        self.use_uad_ch_input = use_uad_ch_input
        self.uad_ip_channels = uad_ip_channels
        self.uad_mode = (self.use_enh_uad or self.use_uad_ch_input)
        self.fpath_uad_masks = fpath_uad_masks
        self.uad_mask_threshold = uad_mask_threshold
        self.uad_mask_path = uad_mask_path
        self.enh_mask_t2 = enh_mask_t2
        self.multi_slice_gt = multi_slice_gt

        '''
        Input and output index specifies which channels of the data is to be used as input
        and output. Data is processed and has a shape of (2, sl, c, x, y). The first dimension
        has the corresponding full-brain and skull-stripped images respectively.

        sl - number of slices
        c - number of contrasts (3 or 4)
        x and y - image matrix dimensions

        For T1-only volumes the c dimension has size 3 where the images are T1-precontrast,
        T1-lowdose contrast and T1-post contrast respectively. For volumes having T2 images, the
        4th dimension has the processed T2 volume.

        For T1-only models, the input_idx is typically [0, 1] and output_idx is [2]
        For T2 models, the input_idx is [0, 1, 3] and output_idx is [2]
        '''

        self.input_idx = input_idx
        self.output_idx = output_idx
        self.enh_pfactor = enh_pfactor
        self.use_enh_mask = use_enh_mask
        self.file_ext = file_ext
        self.uad_file_ext = uad_file_ext
        self.num_channel_output = num_channel_output

        self.ims_cache = ExpiringDict(max_len=250, max_age_seconds=24*60*60)

        self._init_slice_info()

        self.uad_masks = {}
        self._init_uad_masks()
        self._init_img_cache()
        self.csf_quant_dict = {}
        self._init_csf_quant_dict()
        self._init_img_cache()
        self.on_epoch_end()

    def _init_uad_masks(self):
        if self.uad_mode:
            print('Initializing UAD masks...')
            for fpath_mask in tqdm(self.fpath_uad_masks, total=len(self.fpath_uad_masks)):
                dict_key, uad_mask_th = self._process_uad_mask(fpath_mask)
                self.uad_masks[dict_key] = uad_mask_th

    def _init_csf_quant_dict(self):
        if self.enh_mask_t2:
            for fpath in self.data_list:
                fpath_meta = fpath.replace('.', '_meta.')
                meta = load_h5_metadata(fpath_meta)
                dict_key = fpath.split('/')[-1].replace('.{}'.format(self.file_ext), '')
                self.csf_quant_dict[dict_key] = float(meta['t2_csf_quant'])

    def _process_uad_mask(self, fpath_mask):
        dict_key = fpath_mask.split('/')[-1].replace(self.uad_file_ext, '').replace('.', '')

        uad_mask = load_file(fpath_mask, file_type=self.uad_file_ext)

        ### Truncating the uad_mask to min value of 1e-4 to avoid vanishing gradients problem
        ### min_val needs a value of 0.1 if l1_loss has more weightage (e.g. >= 0.6)
        min_val = 1e-2

        th = uad_mask.max() * self.uad_mask_threshold
        uad_mask[uad_mask <= th] = 0
        max_arr = uad_mask.max(axis=(1, 2))

        uad_mask = np.divide(uad_mask, max_arr[:, None, None], where=max_arr[:, None, None]>=min_val)
        uad_mask = np.clip(uad_mask, min_val, uad_mask.max())

        return dict_key, uad_mask[:, None, :, :]

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

    def _init_img_cache(self):
        for fpath in tqdm(self.data_list, total=len(self.data_list)):
            data, data_mask = load_file(fpath, file_type=self.file_ext, params={'h5_key': 'all'})
            self._cache_img(fpath, data, data_mask)

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

    def _fetch_slices_by_dim(self, data, slices, dim):
        if max(slices) >= data.shape[dim]:
            slices = np.clip(slices, min(slices), data.shape[dim]-1)

        if dim == 0:
            ims = data[slices, :, :, :]
        elif dim == 2:
            ims = data[:, :, slices, :]
        elif dim == 3:
            ims = data[:, :, :, slices]

        return ims

    def _get_slices(self, fpath, slices, dim, params={'h5_key': 'all'}):
        cache_cont = self.ims_cache.get(fpath)
        from_cache = False

        if cache_cont is not None:
            data, data_mask = cache_cont
            from_cache = True
        else:
            data, data_mask = load_file(fpath, file_type=self.file_ext, params=params)
            self._cache_img(fpath, data, data_mask)

        ims = self._fetch_slices_by_dim(data, slices, dim)
        ims_mask = self._fetch_slices_by_dim(data_mask, slices, dim)
        return ims, ims_mask

    def _get_uad_mask_slices(self, fpath, slices, dim):
        dict_key = fpath.split('/')[-1].replace('.npy', '').replace('.h5', '')

        if dict_key == 'data':
            ### this happens in inference pipeline
            dict_key = list(self.uad_masks.keys())[0]

        if dict_key not in self.uad_masks:
            fpath_mask = os.path.join(
                self.uad_mask_path, '{}.{}'.format(dict_key, self.uad_file_ext)
            )
            _, uad_mask_th = self._process_uad_mask(fpath_mask)
            self.uad_masks[dict_key] = uad_mask_th

        return self._fetch_slices_by_dim(self.uad_masks[dict_key], slices, dim)

    def _cache_img(self, fpath, ims, ims_mask=None):
        self.ims_cache[fpath] = (ims, ims_mask)

    def _data_generation(self, slice_list_files, slice_list_indexes, enforce_raw_data=False, data_npy=None):
        'Generates data containing batch_size samples'
        # load volumes
        # each element of the data_list contains 3 sets of 3D
        # volumes containing zero, low, and full contrast.
        # the number of slices may differ but the image dimensions
        # should be the same

        t1 = time.time()

        data_list_X = []
        data_list_X_mask = []

        all_slices_X_mask = []

        if not self.predict:
            data_list_Y = []
            data_list_Y_mask = []

        uad_mask_list = []

        for i, (f, idx_dict) in enumerate(zip(slice_list_files, slice_list_indexes)):
            c = idx_dict['index']
            ax = idx_dict['axis']
            csf_quant = 0

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
                slices, slices_mask = self._get_slices(f, idxs, ax, params={'h5_key': 'all'})

            if self.enh_mask_t2:
                dict_key = f.split('/')[-1].replace('.{}'.format(self.file_ext), '')
                csf_quant = self.csf_quant_dict[dict_key]

            uad_mask_slices = None
            if self.uad_mode:
                uad_mask_slices = self._get_uad_mask_slices(f, idxs, ax)

            if ax == 0:
                pass
            if ax == 1:
                assert False, 'invalid slice axis!, {}'.format(ax)
            elif ax == 2:
                slices = np.transpose(slices, (2, 1, 0, 3))
                slices_mask = np.transpose(slices_mask, (2, 1, 0, 3))

                if uad_mask_slices is not None:
                    uad_mask_slices = np.transpose(uad_mask_slices, (2, 1, 0, 3))
            elif ax == 3:
                slices = np.transpose(slices, (3, 1, 0, 2))
                slices_mask = np.transpose(slices_mask, (3, 1, 0, 2))

                if uad_mask_slices is not None:
                    uad_mask_slices = np.transpose(uad_mask_slices, (3, 1, 0, 2))

            if self.resize is not None:
                slices = sp.util.resize(slices, [slices.shape[0], slices.shape[1], self.resize, self.resize])
                slices_mask = sp.util.resize(slices_mask, [slices_mask.shape[0], slices_mask.shape[1], self.resize, self.resize])

                if uad_mask_slices is not None:
                    uad_mask_slices = sp.util.resize(
                        uad_mask_slices, [
                            uad_mask_slices.shape[0], uad_mask_slices.shape[1], self.resize, self.resize
                        ]
                    )

            if self.resample_size is not None:
                if self.verbose > 1:
                    print('Resampling slices to matrix size', self.resample_size)
                slices = resample_slices(slices, self.resample_size)
                slices_mask = resample_slices(slices_mask, self.resample_size)

                if uad_mask_slices:
                    uad_mask_slices = resample_slices(uad_mask_slices, self.resample_size)

            if self.verbose > 1:
                print('loaded slices from {} in {} s'.format(f, time.time() - tic))

            slices_X = slices[:,self.input_idx,:,:][None,...]
            slices_X_mask = slices_mask[:,self.input_idx,:,:][None,...]

            data_list_X.append(slices_X)
            data_list_X_mask.append(slices_X_mask)
            all_slices_X_mask.append(slices_mask)

            if not self.predict:
                if self.multi_slice_gt:
                    slices_Y = slices[:, self.output_idx, :, :][:, None, ...]
                    slices_Y_mask = slices_mask[:, self.output_idx, :, :][:, None, ...]
                else:
                    slices_Y = slices[h, self.output_idx, :, :][None, None, ...]
                    slices_Y_mask = slices_mask[h, self.output_idx, :, :][None, None, ...]

                data_list_Y.append(slices_Y)
                data_list_Y_mask.append(slices_Y_mask)

            if uad_mask_slices is not None:
                uad_mask_list.append(uad_mask_slices)

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
                # FIXME: if percentiles are included in the numpy metadata, then use them as the "max_val_arr" that is passed to the enancement mask (for each sample in the batch)
                # X_mask shape - (batch_size, slices_per_input, num_inputs, height, width)
                # ex: (8, 7, 2, 240, 240)
                # Y_mask shape - (batch_size, 1, 1, height, width)
                # ex: (8, 1, 1, 240, 240)
                if self.enh_mask_t2:
                    enh_mask = get_enh_mask_t2(X_mask, Y_mask, X, center_slice=h, t2_csf_quant=csf_quant)
                else:
                    all_slices_X_mask = np.array(all_slices_X_mask)
                    if len(self.input_idx) == 1:
                        # single contrast model (low-dose as input) - use zero-dose to compute mask
                        x_input = all_slices_X_mask.copy()
                    else:
                        x_input = X_mask
                    enh_mask = enh_mask_smooth(x_input, Y_mask, center_slice=h, p=self.enh_pfactor, multi_slice_gt=self.multi_slice_gt)

            if self.uad_mode:
                uad_mask_list = np.array(uad_mask_list)
                if self.use_enh_uad:
                    if self.multi_slice_gt:
                        enh_mask = uad_mask_list.astype(np.float16)
                    else:
                        enh_mask = uad_mask_list[:, h, ...][:, None, ...].astype(np.float16)

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
        X_mask = np.transpose(np.reshape(X_mask, (X_mask.shape[0], -1, X_mask.shape[3], X_mask.shape[4])), (0, 2, 3, 1))

        if self.use_uad_ch_input:
            x_max = X.max()
            u_min, u_max = uad_mask_list.min(), uad_mask_list.max()

            uad_mask_list = np.interp(uad_mask_list, (u_min, u_max), (u_min, x_max))
            uad_mask_list = uad_mask_list[:, :, 0, ...].transpose(0, 2, 3, 1)

            if self.uad_ip_channels == 1:
                uad_mask_list = uad_mask_list[..., h][..., None]

            uad_mask_list = np.clip(uad_mask_list, 0.1, uad_mask_list.max())

            xs = X.shape
            X = np.reshape(X, (
                xs[0], xs[1], xs[2], self.slices_per_input, len(self.input_idx))
            )
            X_mask = np.reshape(X_mask, (
                xs[0], xs[1], xs[2], self.slices_per_input, len(self.input_idx))
            )

            X = np.append(X, uad_mask_list[..., None], axis=4)
            X_mask = np.append(X_mask, uad_mask_list[..., None], axis=4)

            X = np.reshape(X, (xs[0], xs[1], xs[2], -1))
            X_mask = np.reshape(X_mask, (xs[0], xs[1], xs[2], -1))

        # interleave - tmp code begin
        # X = X.transpose(3, 0, 1, 2)
        # tmp = [None]*(X.shape[0])
        # tmp[::2] = X[:self.slices_per_input]
        # tmp[1::2] = X[self.slices_per_input:]
        # X = np.array(tmp).transpose(1, 2, 3, 0)
        # interleave - tmp code end

        if not self.predict:
            Y = np.transpose(np.reshape(Y, (Y.shape[0], -1, Y.shape[3], Y.shape[4])), (0, 2, 3, 1))
            Y_mask = np.transpose(np.reshape(Y_mask, (Y_mask.shape[0], -1, Y_mask.shape[3], Y_mask.shape[4])), (0, 2, 3, 1))

            if self.multi_slice_gt:
                Y = np.reshape(Y, (self.batch_size, self.slices_per_input, Y.shape[1], Y.shape[2])).transpose(0, 2, 3, 1)
                Y_mask = np.reshape(Y_mask, (self.batch_size, self.slices_per_input, Y_mask.shape[1], Y_mask.shape[2])).transpose(0, 2, 3, 1)

            if self.use_enh_mask or self.use_enh_uad:
                enh_mask = np.transpose(np.reshape(enh_mask, (enh_mask.shape[0], -1, enh_mask.shape[3], enh_mask.shape[4])), (0, 2, 3, 1))
                if self.use_enh_uad:
                    enh_mask = np.interp(enh_mask, (enh_mask.min(), enh_mask.max()), (Y.min(), Y.max()))
            else:
                enh_mask = np.ones(Y.shape)

            Y = np.concatenate((Y, enh_mask), axis=-1)
            Y_mask = np.concatenate((Y_mask, enh_mask), axis=-1)

        if self.verbose > 1:
            print('tranpose data in {} s'.format(time.time() - tic))

        if self.verbose > 1:
            if self.predict:
                print('X, size = ', X.shape)
            else:
                print('X, Y sizes = ', X.shape, Y.shape)
        t2 = time.time()

        if self.predict:
            if self.brain_only:
                return X_mask
            return X
        else:
            if self.brain_only:
                return X_mask, Y_mask
            return X, Y
