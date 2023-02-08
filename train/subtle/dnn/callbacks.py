import os
import numpy as np
import json

import tensorflow as tf
import keras

from subtle.data_loaders import SliceLoader
from subtle.dnn.helpers import make_image, load_data_loader
from subtle.utils.misc import print_progress_bar
from skimage.transform import resize as imresize
import pandas as pd

class TensorBoardImageCallback(keras.callbacks.Callback):
    def __init__(self, model, data_list, slice_dict_list, log_dir, slices_per_epoch=1, slices_per_input=1, batch_size=1, verbose=0, residual_mode=False, max_queue_size=2, num_workers=4, use_multiprocessing=True, shuffle=False, tag='test', gen_type='legacy', positive_only=False, image_index=None, mode='random', input_idx=[0,1], output_idx=[2], resize=None, slice_axis=[0], resample_size=None, brain_only=None, brain_only_mode=None, use_enh_mask=False, enh_pfactor=1.0, model_name=None, block_size=64, block_strides=32, gan_mode=False, detailed_plot=True, plot_list=None, file_ext='npy', uad_mask_path=None, uad_file_ext=None, use_enh_uad=False, use_uad_ch_input=False, uad_ip_channels=1, fpath_uad_masks=[], uad_mask_threshold=0.1, enh_mask_t2=False, multi_slice_gt=False, train_args=None):
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
        self.use_multiprocessing = use_multiprocessing
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
        self.use_uad_ch_input = use_uad_ch_input
        self.uad_ip_channels = uad_ip_channels
        self.detailed_plot = detailed_plot
        self.plot_list = plot_list
        self.tag_list = []
        self.file_ext = file_ext

        self.uad_mask_path = uad_mask_path
        self.use_enh_uad = use_enh_uad
        self.use_uad_ch_input = use_uad_ch_input
        self.fpath_uad_masks = fpath_uad_masks
        self.uad_mask_threshold = uad_mask_threshold
        self.enh_mask_t2 = enh_mask_t2
        self.uad_file_ext = uad_file_ext
        self.multi_slice_gt = multi_slice_gt
        self.train_args = train_args

        self._init_generator()


    def _init_generator(self):
        data_loader = load_data_loader(self.train_args)

        case_nums = [
            c.split('/')[-1].replace(self.file_ext, '').replace('.', '')
            for c in self.data_list
        ]
        fpath_uad_masks = [
            '{}/{}.{}'.format(self.uad_mask_path, cnum, self.uad_file_ext)
            for cnum in case_nums
        ]

        if self.plot_list is not None:
            self.shuffle = False
            self.data_list = list(set([p[0] for p in self.plot_list]))
            self.slice_axis = [0]

        gen_kwargs = {
            'data_list': self.data_list,
            'batch_size': 1,
            'shuffle': self.shuffle,
            'verbose': self.verbose,
            'predict': False,
            'brain_only': self.brain_only,
            'brain_only_mode': self.brain_only_mode,
            'use_enh_uad': self.use_enh_uad,
            'use_uad_ch_input': self.use_uad_ch_input,
            'uad_ip_channels': self.uad_ip_channels,
            'fpath_uad_masks': fpath_uad_masks,
            'uad_mask_threshold': self.uad_mask_threshold,
            'uad_mask_path': self.uad_mask_path,
            'uad_file_ext': self.uad_file_ext,
            'enh_mask_t2': self.enh_mask_t2
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
                'file_ext': self.file_ext
            }

        gen_kwargs = {**gen_kwargs, **kw}
        self.generator =  data_loader(**gen_kwargs)

        if self.plot_list is not None:
            self.img_indices = np.array([
                np.where(self.generator.slice_list_files == fpath_case)[0][idx]
                for fpath_case, idx in self.plot_list
            ])

            self.tag_list = [
                '{}_{}'.format(fpath_case.split('/')[-1].replace('.h5', ''), idx + 1)
                for fpath_case, idx in self.plot_list
            ]
        else:
            self.img_indices = np.random.choice(range(self.generator.__len__()), size=self.batch_size, replace=False)

    def on_epoch_end(self, epoch, logs={}):
        #_len = self.generator.__len__()
        writer = tf.summary.FileWriter(self.log_dir)
        for idx, ii in enumerate(self.img_indices):
            if len(self.tag_list) == 0:
                tag = '{}_{}'.format(self.tag, idx)
            else:
                tag = self.tag_list[idx]

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

                input_len = len(self.input_idx)

                if self.use_uad_ch_input:
                    input_len += 1

                X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], self.slices_per_input, input_len))

                h = self.slices_per_input // 2
                X_center = X[...,h,:] # [1, nx, ny, N]

                if self.gen_type == 'legacy' and self.residual_mode and len(self.input_idx) == 2:
                    Y_prediction = X_center[..., 0] + Y_prediction
                    Y = X_center[..., 0] + Y
                    X_center[..., 1] = X_center[..., 1] + X_center[..., 0]

                if self.detailed_plot:
                    imgs = (X_center, Y, Y_prediction)
                else:
                    y_disp = np.array([Y[..., 0]]).transpose(1, 2, 3, 0)
                    imgs = (y_disp, Y_prediction)

                display_image = np.concatenate((imgs), axis=3).transpose((0, 1, 3, 2)).reshape((X_center.shape[1], -1))

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

    def on_train_end(self, logs=None):
        logfile = self._get_fhandle()
        print('done training', file=logfile)
        logfile.close()

class HparamsCallback(keras.callbacks.TensorBoard):
    def __init__(self, log_dir, tunable_args, **kwargs):
        super(HparamsCallback, self).__init__(log_dir, **kwargs)

        self.log_dir = log_dir
        self.tunable_args = tunable_args
        self.trial_id = self.log_dir.split('/')[-2].split('_')[1]

    def on_train_begin(self, logs=None):
        # write to CSV for hypmonitor API
        df_params = pd.DataFrame(list(self.tunable_args.items()), columns=['Hyperparam', 'Value'])
        df_params.to_csv(os.path.join(self.log_dir, '..', 'params.csv'))

        keras.callbacks.TensorBoard.on_train_begin(self, logs=logs)

        exp_id = self.log_dir.split('/')[-3]
        hypmonitor_port = str(os.environ['HYPMONITOR_PORT'])

        # Markdown formatting commented because app environment supports only python 3.5 for now; formatting requires python 3.6+

        # disp = f'''### Hyperparameter Summary [Detailed logs](http://localhost:{hypmonitor_port}/experiment?id={exp_id})\n'''
        # disp += f'''| *Hyperparameter* | *Value* |\n'''
        # disp += f'''| --------------- | ------- |\n'''
        #
        # for k, v in self.tunable_args.items():
        #     v = '{:.3f}'.format(v)
        #     disp += f'''| {k} | {v} |\n'''

        disp = 'Hyperparameter Summary (Detailed logs - http://localhost:{}/experiment?id={}))\n'.format(hypmonitor_port, exp_id)

        disp += '| Hyperparameter | Value |\n'
        disp += '| -------------- | ----- |\n'

        for k, v in self.tunable_args.items():
            v = '{:.3f}'.format(v)
            disp += '| {} | {} |\n'.format(k, v)

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
