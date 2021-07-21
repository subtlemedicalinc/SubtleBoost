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
from subtle.utils.experiment import get_model_config, get_layer_config
from subtle.dnn.callbacks import TensorBoardCallBack, TensorBoardImageCallback, TrainProgressCallBack, HparamsCallback
import pdb

class GeneratorBase:
    def __init__(
        self, num_channel_input=1, num_channel_output=1, img_rows=128, img_cols=128, img_depth=128, optimizer_fun=Adam, lr_init=None, optim_amsgrad=True, loss_function=suloss.l1_loss, metrics_monitor=[suloss.l1_loss], verbose=True, checkpoint_file=None, log_dir=None, job_id='', save_best_only=True, compile_model=True,
        model_config='base', tunable_params=None, fpaths_pre=[], transfer_weights=True
    ):
        self.num_channel_input = num_channel_input
        self.num_channel_output = num_channel_output
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_depth = img_depth
        self.optimizer_fun = optimizer_fun
        self.lr_init = lr_init
        self.loss_function = loss_function
        self.metrics_monitor = metrics_monitor
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file
        self.log_dir = log_dir
        self.job_id = job_id
        self.save_best_only = save_best_only
        self.compile_model = compile_model
        self.optim_amsgrad = optim_amsgrad
        self.model_config = model_config
        self.tunable_params = tunable_params
        self.fpaths_pre = fpaths_pre
        self.transfer_weights = transfer_weights

        self.model = None # to be assigned by _build_model() in children classes

        self._init_model_config()

    def _init_model_config(self):
        self.config_dict = get_model_config(self.model_name, self.model_config, model_type='generators', dirpath_config='/home/jiang/projects/SubtleGad/train/configs/models')

        if self.tunable_params:
            self.config_dict = {**self.config_dict, **self.tunable_params}

        for k, v in self.config_dict.items():
            if not isinstance(v, dict):
                # attributes like self.num_conv_per_pooling are assigned here
                setattr(self, k, v)

    def callback_checkpoint(self, filename=None, monitor='val_l1_loss'):
        if filename is not None:
            self.checkpoint_file = filename

        return keras.callbacks.ModelCheckpoint(self.checkpoint_file, monitor=monitor, save_best_only=self.save_best_only)

    def callback_tensorboard(self, log_dir=None, log_every=None):
        if log_dir is None:
            _log_dir = self.log_dir
        else:
            _log_dir = log_dir

        if log_every is not None and log_every > 0:
            return TensorBoardCallBack(log_every=log_every, log_dir=_log_dir, batch_size=8, write_graph=False)
        else:
            return keras.callbacks.TensorBoard(log_dir=_log_dir, batch_size=8, write_graph=False)

    def callback_progress(self, log_dir, data_loader):
        if not log_dir:
            log_dir = self.log_dir

        return TrainProgressCallBack(log_dir=log_dir, data_loader=data_loader)

    def callback_hparams(self, log_dir, tunable_args):
        if not log_dir:
            log_dir = self.log_dir

        return HparamsCallback(log_dir=log_dir, tunable_args=tunable_args)

    def callback_csv(self, fpath_csv):
        return keras.callbacks.CSVLogger(fpath_csv, append=True)

    def callback_tbimage(self, data_list, slice_dict_list, slices_per_epoch=1, slices_per_input=1, batch_size=1, verbose=0, residual_mode=False, max_queue_size=2, num_workers=4, use_multiprocessing=True, tag='test', gen_type='legacy', log_dir=None, shuffle=False, image_index=None, input_idx=[0,1], output_idx=[2], slice_axis=0, resize=None, resample_size=None, brain_only=None, brain_only_mode=None, model_name=None, block_size=64, block_strides=16, gan_mode=False, use_enh_mask=False, enh_pfactor=1.0, detailed_plot=True, plot_list=None, file_ext=None, uad_mask_path=None, uad_ip_channels=1, uad_file_ext=None, use_enh_uad=False, use_uad_ch_input=False, uad_mask_threshold=0.1, enh_mask_t2=False, multi_slice_gt=False):
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
                brain_only_mode=brain_only_mode,
                model_name=model_name,
                block_size=block_size,
                block_strides=block_strides,
                gan_mode=gan_mode,
                use_enh_mask=use_enh_mask,
                enh_pfactor=enh_pfactor,
                detailed_plot=detailed_plot,
                plot_list=plot_list,
                file_ext=file_ext,
                uad_mask_path=uad_mask_path,
                use_enh_uad=use_enh_uad,
                use_uad_ch_input=use_uad_ch_input,
                uad_ip_channels=uad_ip_channels,
                uad_mask_threshold=uad_mask_threshold,
                enh_mask_t2=enh_mask_t2,
                uad_file_ext=uad_file_ext,
                multi_slice_gt=multi_slice_gt
            )

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

    def get_config(self, param_name, layer_name=''):
        return get_layer_config(self.config_dict, param_name, layer_name)

    def _freeze_weights(self, kw=None):
        if kw is not None:
            layers = [l for l in self.model.layers if kw in l.name]
        else:
            layers = self.model.layers

        for layer in layers:
            layer.trainable = False

    def _compile_model(self, custom_optim=None):
        if custom_optim is not None:
            optimizer = custom_optim
        elif self.lr_init is not None:
            optimizer = self.optimizer_fun(lr=self.lr_init, amsgrad=self.optim_amsgrad, clipnorm=1)
        else:
            optimizer = self.optimizer_fun()

        self.model.compile(loss=self.loss_function, optimizer=optimizer, metrics=self.metrics_monitor)
