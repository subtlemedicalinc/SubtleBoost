import numpy as np

import tensorflow as tf
import keras

from subtle.data_loaders import SliceLoader
from subtle.dnn.helpers import make_image

class TensorBoardImageCallback(keras.callbacks.Callback):
    def __init__(self, model, data_list, slice_dict_list, log_dir, slices_per_epoch=1, slices_per_input=1, batch_size=1, verbose=0, residual_mode=False, max_queue_size=2, num_workers=4, use_multiprocessing=True, shuffle=False, tag='test', gen_type='legacy', positive_only=False, image_index=None, mode='random', input_idx=[0,1], output_idx=[2], resize=None, slice_axis=[0], resample_size=None, brain_only=None, brain_only_mode=None):
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

        self._init_generator()


    def _init_generator(self):
            self.generator =  SliceLoader(data_list=self.data_list,
                    batch_size=1,
                    shuffle=self.shuffle,
                    verbose=self.verbose,
                    residual_mode=self.residual_mode,
                    positive_only = self.positive_only,
                    slices_per_input=self.slices_per_input,
                    input_idx=self.input_idx,
                    output_idx=self.output_idx,
                    predict=False,
                    resize=self.resize,
                    resample_size=self.resample_size,
                    slice_axis=self.slice_axis,
                    brain_only=self.brain_only,
                    brain_only_mode=self.brain_only_mode)

            self.img_indices = np.random.choice(range(self.generator.__len__()), size=self.batch_size, replace=False)

    def on_epoch_end(self, epoch, logs={}):
        #_len = self.generator.__len__()
        writer = tf.summary.FileWriter(self.log_dir)
        for ii in self.img_indices:
            tag = '{}_{}'.format(self.tag, ii)
            # X is [1, nx, ny, N * 2.5d]
            # Y is [1, nx, ny, N]

            raw_data=False
            # enforce_raw_data=(ii >= 5) [uncomment when using masked + full head mixed model]

            X, Y = self.generator.__getitem__(ii, enforce_raw_data=raw_data)
            Y_prediction = self.model.predict_on_batch(X)

            if self.generator._current_file_list:
                current_fpath = self.generator._current_file_list[0]
                case_num = current_fpath.split('/')[-1].replace('.h5', '')
                slice_idx = self.generator._current_slice_list[0]
                tag = '{} ({}: slice={}_{})'.format(tag, case_num, slice_idx['axis'], slice_idx['index'])

            #print(X.shape, Y.shape, Y_prediction.shape)
            # separate 2.5D and N
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


class TensorBoardCallBack(keras.callbacks.TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter+=1
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
