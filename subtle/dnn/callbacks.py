import os
import numpy as np
import json

import tensorflow as tf
import keras

from subtle.data_loaders import SliceLoader
from subtle.dnn.helpers import make_image, load_data_loader
from subtle.subtle_io import print_progress_bar
from scipy.misc import imresize

class TensorBoardImageCallback(keras.callbacks.Callback):
    def __init__(self, model, data_list, slice_dict_list, log_dir, slices_per_epoch=1, slices_per_input=1, batch_size=1, verbose=0, residual_mode=False, max_queue_size=2, num_workers=4, use_multiprocessing=True, shuffle=False, tag='test', gen_type='legacy', positive_only=False, image_index=None, mode='random', input_idx=[0,1], output_idx=[2], resize=None, slice_axis=[0], resample_size=None, brain_only=None, brain_only_mode=None, use_enh_mask=False, enh_pfactor=1.0, model_name=None, block_size=64, block_strides=32, gan_mode=False):
        super().__init__()
        self.tag = tag
        self.data_list = data_list
        self.slice_dict_list = slice_dict_list
        self.slices_per_epoch = slices_per_epoch
        self.slices_per_input = slices_per_input
        self.batch_size = batch_size
        self.verbose = verbose
        self.residual_mode = residual_mode
        self.model = model
        self.log_dir = log_dir
        self.max_queue_size  = max_queue_size
        self.num_workers = num_workers
        self.use_multiprocessing=use_multiprocessing
        self.shuffle = shuffle
        self.gen_type = gen_type
        self.positive_only = positive_only
        self.image_index = image_index
        self.mode = mode
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.resize = resize
        self.slice_axis = slice_axis
        self.resample_size = resample_size
        self.brain_only = brain_only
        self.brain_only_mode = brain_only_mode
        self.enh_pfactor = enh_pfactor
        self.model_name = model_name
        self.block_size = block_size
        self.block_strides = block_strides
        self.gan_mode = gan_mode
        self.use_enh_mask = use_enh_mask

        self._init_generator()


    def _init_generator(self):
        data_loader = load_data_loader(self.model_name)
        gen_kwargs = {
            'data_list': self.data_list,
            'batch_size': 1,
            'shuffle': self.shuffle,
            'verbose': self.verbose,
            'predict': False,
            'brain_only': self.brain_only,
            'brain_only_mode': self.brain_only_mode
        }

        if '3d' in self.model_name:
            kw = {
                'block_size': self.block_size,
                'block_strides': self.block_strides
            }
        else:
            kw = {
                'residual_mode': self.residual_mode,
                'positive_only':  self.positive_only,
                'slices_per_input': self.slices_per_input,
                'input_idx': self.input_idx,
                'output_idx': self.output_idx,
                'slice_axis': self.slice_axis,
                'resize': self.resize,
                'resample_size': self.resample_size,
                'use_enh_mask': self.use_enh_mask,
                'enh_pfactor': self.enh_pfactor,
            }

        gen_kwargs = {**gen_kwargs, **kw}
        self.generator =  data_loader(**gen_kwargs)

        self.img_indices = np.random.choice(range(self.generator.__len__()), size=self.batch_size, replace=False)

    def on_epoch_end(self, epoch, logs={}):
        #_len = self.generator.__len__()
        writer = tf.summary.FileWriter(self.log_dir)
        for idx, ii in enumerate(self.img_indices):
            tag = '{}_{}'.format(self.tag, idx)
            # X is [1, nx, ny, N * 2.5d]
            # Y is [1, nx, ny, N]

            raw_data=False
            # enforce_raw_data=(ii >= 5) [uncomment when using masked + full head mixed model]

            if '3d' in self.model_name:
                X, Y, _ = self.generator.__getitem__(ii, enforce_raw_data=raw_data)
                Y_prediction = self.model.predict_on_batch(X)

                slice_idx = (idx + 1) % 3

                if slice_idx == 0:
                    pass
                elif slice_idx == 1:
                    X = X.transpose(0, 2, 1, 3, 4)
                    Y = Y.transpose(0, 2, 1, 3, 4)
                    Y_prediction = Y_prediction.transpose(0, 2, 1, 3, 4)
                elif slice_idx == 2:
                    X = X.transpose(0, 1, 3, 2, 4)
                    Y = Y.transpose(0, 1, 3, 2, 4)
                    Y_prediction = Y_prediction.transpose(0, 1, 3, 2, 4)

                pidx = X.shape[1] // 2
                X_zero = X[0, pidx, :, :, 0]
                X_low = X[0, pidx, :, :, 1]
                Y_full = Y[0, pidx, :, :, 0]
                Y_pred_full = Y_prediction[0, pidx, :, :, 0]

                display_image = np.hstack([X_zero, X_low, Y_full, Y_pred_full])
                display_image = imresize(display_image, (display_image.shape[0]*3, display_image.shape[1]*3))
            else:
                X, Y = self.generator.__getitem__(ii, enforce_raw_data=raw_data)
                Y_prediction = self.model.predict_on_batch(X)

                if self.gan_mode:
                    Y_prediction = Y_prediction[0]

                X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], self.slices_per_input, len(self.input_idx)))

                h = self.slices_per_input // 2
                X_center = X[...,h,:] # [1, nx, ny, N]

                if self.gen_type == 'legacy' and self.residual_mode and len(self.input_idx) == 2:
                    Y_prediction = X_center[..., 0] + Y_prediction
                    Y = X_center[..., 0] + Y
                    X_center[..., 1] = X_center[..., 1] + X_center[..., 0]

                display_image = np.concatenate((X_center, Y, Y_prediction), axis=3).transpose((0,1,3,2)).reshape((X_center.shape[1], -1))

            image = make_image(display_image)
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=image)])
            writer.add_summary(summary, epoch)
        writer.close()

        return

class TrainProgressCallBack(keras.callbacks.Callback):
    def __init__(self, log_dir, data_loader, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.data_loader = data_loader

        parent_dir = self.log_dir.split('/')[-1]
        self.fpath_log = os.path.join(self.log_dir, '..', 'log_train_{}.log'.format(parent_dir))
        self.total = self.data_loader.__len__()

    def _get_fhandle(self):
        fmode = 'a' if os.path.exists(self.fpath_log) else 'w'
        return open(self.fpath_log, fmode)

    def _get_metric_str(self, metrics):
        exclude_keys = ['batch', 'size']
        return ' '.join([
            '{}: {:.3f}'.format(k, v)
            for k, v in metrics.items()
            if k not in exclude_keys
        ])

    def on_epoch_begin(self, epoch, logs={}):
        logfile = self._get_fhandle()
        print('Epoch #{}\n'.format(epoch + 1), file=logfile)

    def on_batch_end(self, batch, logs={}):
        logfile = self._get_fhandle()

        metrics = self._get_metric_str(logs)
        print_progress_bar(batch, self.total, logfile, prefix='Training', suffix=metrics)

    def on_epoch_end(self, epoch, logs={}):
        logfile = self._get_fhandle()
        metrics = self._get_metric_str(logs)
        print('End of epoch #{} - {}\n'.format(epoch + 1, metrics), file=logfile)
        logfile.close()

class HparamsCallback(keras.callbacks.TensorBoard):
    def __init__(self, log_dir, tunable_args, **kwargs):
        super(HparamsCallback, self).__init__(log_dir, **kwargs)

        self.tunable_args = tunable_args
        self.trial_id = self.log_dir.split('/')[-2].split('_')[1]

    def on_train_begin(self, logs=None):
        keras.callbacks.TensorBoard.on_train_begin(self, logs=logs)

        disp = f'''### Hyperparameter Summary\n'''
        disp += f'''| *Hyperparameter* | *Value* |\n'''
        disp += f'''| --------------- | ------- |\n'''

        for k, v in self.tunable_args.items():
            v = '{:.3f}'.format(v)
            disp += f'''| {k} | {v} |\n'''

        tensor =  tf.convert_to_tensor(disp)
        summary = tf.summary.text("Trial {}".format(self.trial_id), tensor)

        with tf.Session() as sess:
            s = sess.run(summary)
            self.writer.add_summary(s)

class TensorBoardCallBack(keras.callbacks.TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter%self.log_every==0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)

def plot_tb(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
