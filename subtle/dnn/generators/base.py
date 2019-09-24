'''
Network architecture for SubtleGad project.

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/25
'''

import tensorflow as tf
import keras.models
import keras.callbacks
from keras.optimizers import Adam

from warnings import warn
import numpy as np

import subtle.subtle_loss as suloss
from subtle.dnn.callbacks import TensorBoardCallBack, TensorBoardImageCallback

# based on u-net and v-net
class GeneratorBase:
    def __init__(self,
            num_channel_input=1, num_channel_output=1, img_rows=128, img_cols=128, img_depth=128,
            num_channel_first=32, optimizer_fun=Adam, final_activation='linear',
            lr_init=None, loss_function=suloss.l1_loss,
            metrics_monitor=[suloss.l1_loss],
            num_poolings=3, num_conv_per_pooling=3,
            batch_norm=True, verbose=True, checkpoint_file=None, log_dir=None, job_id='', save_best_only=True, compile_model=True):

        self.num_channel_input = num_channel_input
        self.num_channel_output = num_channel_output
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_depth = img_depth
        self.num_channel_first = num_channel_first
        self.optimizer_fun = optimizer_fun
        self.final_activation = final_activation
        self.lr_init = lr_init
        self.loss_function = loss_function
        self.metrics_monitor = metrics_monitor
        self.num_poolings = num_poolings
        self.num_conv_per_pooling = num_conv_per_pooling
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file
        self.log_dir = log_dir
        self.job_id = job_id
        self.save_best_only = save_best_only
        self.compile_model = compile_model

        self.model = None # to be assigned by _build_model() in children classes

    def callback_checkpoint(self, filename=None):
        if filename is not None:
            self.checkpoint_file = filename

        return keras.callbacks.ModelCheckpoint(self.checkpoint_file, monitor='val_loss', save_best_only=self.save_best_only)

    def callback_tensorbaord(self, log_dir=None, log_every=None):
        if log_dir is None:
            _log_dir = self.log_dir
        else:
            _log_dir = log_dir

        if log_every is not None and log_every > 0:
            return TensorBoardCallBack(log_every=log_every, log_dir=_log_dir, batch_size=8, write_graph=False)
        else:
            return keras.callbacks.TensorBoard(log_dir=_log_dir, batch_size=8, write_graph=False)

    def callback_tbimage(self, data_list, slice_dict_list, slices_per_epoch=1, slices_per_input=1, batch_size=1, verbose=0, residual_mode=False, max_queue_size=2, num_workers=4, use_multiprocessing=True, tag='test', gen_type='legacy', log_dir=None, shuffle=False, image_index=None, input_idx=[0,1], output_idx=[2], slice_axis=0, resize=None, resample_size=None, brain_only=None, brain_only_mode=None):
        if log_dir is None:
            _log_dir = self.log_dir
        else:
            _log_dir = log_dir
        return TensorBoardImageCallback(self,
                data_list=data_list,
                slice_dict_list=slice_dict_list,
                log_dir=_log_dir,
                slices_per_epoch=slices_per_epoch,
                slices_per_input=slices_per_input,
                batch_size=batch_size,
                verbose=verbose,
                residual_mode=residual_mode,
                max_queue_size=max_queue_size,
                num_workers=num_workers,
                use_multiprocessing=use_multiprocessing,
                shuffle=shuffle,
                tag=tag,
                gen_type=gen_type,
                image_index=image_index,
                input_idx=input_idx,
                output_idx=output_idx,
                slice_axis=slice_axis,
                resize=resize,
                resample_size=resample_size,
                brain_only=brain_only,
                brain_only_mode=brain_only_mode)

    def load_weights(self, filename=None):
        if filename is not None:
            self.checkpoint_file = filename
        try:
            if self.verbose:
                print('loading weights from', self.checkpoint_file)
            self.model.load_weights(self.checkpoint_file)
        except Exception as e:
            warn('failed to load weights. training from scratch')
            warn(str(e))

    def _compile_model(self):
        if self.lr_init is not None:
            optimizer = self.optimizer_fun(lr=self.lr_init, amsgrad=True)#,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        else:
            optimizer = self.optimizer_fun()

        self.model.compile(loss=self.loss_function, optimizer=optimizer, metrics=self.metrics_monitor)
