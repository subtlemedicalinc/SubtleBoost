'''
subtle_gad_network.py

Network architecture for SubtleGad project.

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/25
'''

import tensorflow as tf
import keras.models
from keras.layers import Input, merge, Conv2D, Conv2DTranspose, BatchNormalization, Convolution2D, MaxPooling2D, UpSampling2D, Dense, concatenate
import keras.callbacks
from keras.layers.merge import add as keras_add
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, mean_squared_error

from warnings import warn
import numpy as np

import os
import time

import subtle.subtle_io as suio

# clean up
def clear_keras_memory():
    keras.backend.clear_session()

# use part of memory
def set_keras_memory(limit=0.9):
    from tensorflow import ConfigProto as tf_ConfigProto
    from tensorflow import Session as tf_Session
    from keras.backend.tensorflow_backend import set_session
    config = tf_ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = limit
    set_session(tf_Session(config=config))

def make_image(im):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    import imageio
    import io
    nx, ny = im.shape
    im_uint = im.astype(np.uint8)
    output = io.BytesIO()
    imageio.imwrite(output, im, format='png')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=nx,
                         width=ny,
                         encoded_image_string=image_string)

class TensorBoardImageCallback(keras.callbacks.Callback):
    def __init__(self, model, data_list, slice_dict_list, log_dir, slices_per_epoch=1, slices_per_input=1, batch_size=1, verbose=0, residual_mode=False,
            max_queue_size=2, num_workers=4, use_multiprocessing=True, shuffle=False, tag='test'):
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

        self.batch_size = 1

        self._init_generator()


    def _init_generator(self):
        self.generator =  suio.DataGenerator(data_list=self.data_list,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                verbose=self.verbose, 
                residual_mode=self.residual_mode,
                slices_per_input=self.slices_per_input,
                predict=True)

    def on_epoch_end(self, epoch, logs={}):
        _len = self.generator.__len__()
        X = self.generator.__getitem__(_len // 2)
        #Y_prediction = self.model.predict_generator(generator=self.generator, steps=1, max_queue_size=self.max_queue_size, workers=self.num_workers, use_multiprocessing=self.use_multiprocessing, verbose=self.verbose)
        Y_prediction = self.model.predict_on_batch(X)
        if self.residual_mode:
            h = self.slices_per_input // 2
            Y_prediction = X[:,:,:,h].squeeze() + Y_prediction.squeeze()
        image = make_image(Y_prediction.squeeze())
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter(self.log_dir)
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

# based on u-net and v-net
class DeepEncoderDecoder2D:
    def __init__(self,
            num_channel_input=1, num_channel_output=1, img_rows=128, img_cols=128, 
            num_channel_first=32, optimizer_fun=Adam, final_activation='linear',
            lr_init=None, loss_function=mean_absolute_error,
            #metrics_monitor=[PSNRLoss, mean_absolute_error, mean_squared_error],
            metrics_monitor=[mean_absolute_error, mean_squared_error],
            num_poolings=3, num_conv_per_pooling=3,
            batch_norm=True, verbose=True, checkpoint_file=None, log_dir=None, job_id=''):

        self.num_channel_input = num_channel_input
        self.num_channel_output = num_channel_output
        self.img_rows = img_rows
        self.img_cols = img_cols
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

        self.model = None # to be assigned by _build_model()
        self._build_model()

    #def get_model(self):
        #return self.model

    def callback_checkpoint(self, filename=None):
        
        if filename is not None:
            self.checkpoint_file = filename
        
        return keras.callbacks.ModelCheckpoint(self.checkpoint_file, monitor='val_loss', save_best_only=True)

    # FIXME check
    def callback_tensorbaord(self, log_dir=None, log_every=None):

        if log_dir is not None:
            self.log_dir = log_dir
        
        if log_every is not None and log_every > 0:
            return TensorBoardCallBack(log_every=log_every, log_dir=os.path.join(self.log_dir, '{}_{}'.format(self.job_id, time.time())), batch_size=8, write_graph=False)
        else:
            return keras.callbacks.TensorBoard(log_dir=os.path.join(self.log_dir, '{}_{}'.format(self.job_id, time.time())), batch_size=8, write_graph=False)

    # FIXME check
    def callback_tbimage(self, data_list, slice_dict_list, slices_per_epoch=1, slices_per_input=1, batch_size=1, verbose=0, residual_mode=False, 
            max_queue_size=2, num_workers=4, use_multiprocessing=True, tag='test'):
        return TensorBoardImageCallback(self,
                data_list=data_list,
                slice_dict_list=slice_dict_list,
                log_dir=os.path.join(self.log_dir, '{}_{}'.format(self.job_id, time.time())),
                slices_per_epoch=slices_per_epoch,
                slices_per_input=slices_per_input,
                batch_size=batch_size,
                verbose=verbose,
                residual_mode=residual_mode,
                max_queue_size=max_queue_size,
                num_workers=num_workers,
                use_multiprocessing=use_multiprocessing,
                tag=tag)

    def load_weights(self, filename=None):
        if filename is not None:
            self.checkpoint_file = filename
        try:
            print('loading weights from', self.checkpoint_file)
            self.model.load_weights(self.checkpoint_file)
        except Exception as e:
            warn('failed to load weights. training from scratch')
            warn(str(e))

    def _build_model(self):

        # batch norm
        if self.batch_norm:
            lambda_bn = lambda x: BatchNormalization()(x)
        else:
            lambda_bn = lambda x: x

        # layers
        # 2D input is (rows, cols, channels)

        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input))  

        if self.verbose:
            print(inputs)

        # step 1
        conv1 = inputs

        for i in range(self.num_conv_per_pooling):

            conv1 = Conv2D(filters=self.num_channel_first, kernel_size=(3, 3), padding="same", activation="relu")(conv1)
            conv1 = lambda_bn(conv1)    

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        if self.verbose:
            print(conv1, pool1)

        # encoder pools
        convs = [inputs, conv1]
        pools = [inputs, pool1]
        list_num_features = [self.num_channel_input, self.num_channel_first]

        for i in range(1, self.num_poolings):

            # step 2
            conv_encoder = pools[-1]
            num_channel = self.num_channel_first * (2**i) # double channels each step
            # FIXME: check if this should be 2**i ?

            for j in range(self.num_conv_per_pooling):

                conv_encoder = Conv2D(filters=num_channel, kernel_size=(3, 3), padding="same", activation="relu")(conv_encoder)
                conv_encoder = lambda_bn(conv_encoder)    

            pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)

            if self.verbose:
                print(conv_encoder, pool_encoder)

            pools.append(pool_encoder)
            convs.append(conv_encoder)
            list_num_features.append(num_channel)

        # center connection
        conv_center = Conv2D(filters=list_num_features[-1], kernel_size=(3, 3), padding="same", activation="relu",
                kernel_initializer='zeros',
                bias_initializer='zeros')(pools[-1])     

        # residual connection
        conv_center = keras_add([pools[-1], conv_center])

        if self.verbose:
                print(conv_center)

        # decoder steps
        conv_decoders = [conv_center]

        for i in range(1, self.num_poolings + 1):

            #print('decoder', i, convs, pools)
            #print(UpSampling2D(size=(2, 2))(conv_center))
            #print(convs[-i])

            up_decoder = concatenate([UpSampling2D(size=(2, 2))(conv_decoders[-1]), convs[-i]])
            conv_decoder = up_decoder

            for j in range(self.num_conv_per_pooling):

                conv_decoder = Conv2D(filters=list_num_features[-i], kernel_size=(3, 3),
                        padding="same", activation="relu")(conv_decoder)
                conv_decoder = lambda_bn(conv_decoder)     

            conv_decoders.append(conv_decoder)

            if self.verbose:
                    print(conv_decoder, up_decoder)        

        # output layer

        conv_decoder = conv_decoders[-1]

        conv_output = Conv2D(self.num_channel_output, (1, 1), padding="same", activation=self.final_activation)(conv_decoder)    

        if self.verbose:
            print(conv_output)
        
        # model
        model = keras.models.Model(inputs=inputs, outputs=conv_output)

        if self.verbose:
            print(model)
        
        # fit
        if self.lr_init is not None:
            optimizer = self.optimizer_fun(lr=self.lr_init, amsgrad=True)#,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        else:
            optimizer = self.optimizer_fun()
        model.compile(loss=self.loss_function, optimizer=optimizer, metrics=self.metrics_monitor)
        
        self.model = model
