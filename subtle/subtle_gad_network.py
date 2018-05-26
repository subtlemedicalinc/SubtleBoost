'''
subtle_gad_network.py

Network architecture for SubtleGad project.

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/25
'''

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge, Conv2D, Conv2DTranspose, BatchNormalization, Convolution2D, MaxPooling2D, UpSampling2D, Dense, concatenate
from keras.layers.merge import add as keras_add
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, mean_squared_error
from keras import backend as K
#from cafndl_metrics import PSNRLoss

import numpy as np


# clean up
def clear_keras_memory():
    ks.backend.clear_session()

# use part of memory
def setKerasMemory(limit=0.3):
    from tensorflow import ConfigProto as tf_ConfigProto
    from tensorflow import Session as tf_Session
    from keras.backend.tensorflow_backend import set_session
    config = tf_ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = limit
    set_session(tf_Session(config=config))

# based on u-net and v-net
class DeepEncoderDecoder2D:
    def __init__(self,
            num_channel_input=1, num_channel_output=1, img_rows=128, img_cols=128, 
            num_channel_first=32, optimizer_fun=Adam, y=np.array([-1, 1]),
            lr_init=None, loss_function=mean_absolute_error,
            #metrics_monitor=[PSNRLoss, mean_absolute_error, mean_squared_error],
            metrics_monitor=[mean_absolute_error, mean_squared_error],
            num_poolings=3, num_conv_per_pooling=3,
            with_bn=False, verbose=True):

        self.num_channel_input = num_channel_input
        self.num_channel_output = num_channel_output
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_channel_first = num_channel_first
        self.optimizer_fun = optimizer_fun
        self.y = y
        self.lr_init = lr_init
        self.loss_function = loss_function
        self.metrics_monitor = metrics_monitor
        self.num_poolings = num_poolings
        self.num_conv_per_pooling = num_conv_per_pooling
        self.with_bn = with_bn
        self.verbose = verbose

        self.model = None # to be assigned by _build_model()
        self._build_model()

    #def get_model(self):
        #return self.model

    def _build_model(self):

        # batch norm
        if self.with_bn:
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

        if np.max(np.abs(self.y)) <= 1:
            if np.min(np.array(self.y)) < 0:
                #tanh -1~+1
                conv_output = Conv2D(self.num_channel_output, (1, 1), padding="same", activation="tanh")(conv_decoder)
                print('use tanh activation')
            else:
                conv_output = Conv2D(self.num_channel_output, (1, 1), padding="same", activation='sigmoid')(conv_decoder)    
                print('use sigmoid activation')
        else:
            conv_output = Conv2D(self.num_channel_output, (1, 1), padding="same", activation='linear')(conv_decoder)    
            print('use linear activation')

        if self.verbose:
            print(conv_output)
        
        # model
        model = Model(inputs=inputs, outputs=conv_output)

        if self.verbose:
            print(model)
        
        # fit
        if self.lr_init is not None:
            optimizer = self.optimizer_fun(lr=self.lr_init)#,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        else:
            optimizer = self.optimizer_fun()
        model.compile(loss=self.loss_function, optimizer=optimizer, metrics=self.metrics_monitor)
        
        self.model = model