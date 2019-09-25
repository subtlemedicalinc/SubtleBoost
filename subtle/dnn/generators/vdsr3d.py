"""
Based on VDSR implementation in SubtleMR by Long Wang
"""

import tensorflow as tf
import keras.models
from keras.layers import Input, Conv3D, Activation
from keras.layers.merge import add as keras_add
from keras.initializers import he_normal

from subtle.dnn.generators.base import GeneratorBase
from subtle.dnn.layers.ConstMultiplier import ConstMultiplier


class GeneratorVDSR3D(GeneratorBase):
    def __init__(self, num_layers=20, scale_factor=0.1, init_seed=15213, **kwargs):
        super().__init__(**kwargs)
        self.init_seed = init_seed
        self.scale_factor = scale_factor
        self.num_layers = num_layers

        self._build_model()

    def _conv(self, x, features=None):
        features = self.num_channel_first if features is None else features
        return Conv3D(
            features, kernel_size=3, strides=1, activation=None,
            kernel_initializer=he_normal(seed=self.init_seed),
            padding='same'
        )(x)

    def _resblock(self, x, act='relu'):
        res_conv1 = Activation(act)(self._conv(x))
        return self._conv(res_conv1)


    def _build_model(self):
        print('Building VDSR 3D model...')
        inputs = Input(shape=(self.img_rows, self.img_cols, self.img_depth, self.num_channel_input))

        if self.verbose:
            print(inputs)

        conv = inputs
        for _ in range(self.num_layers-1):
            conv = Activation('relu')(self._conv(conv))

        conv = Activation('tanh')(self._conv(conv, features=1))
        conv_output = keras_add([inputs, conv])

        # model
        model = keras.models.Model(inputs=inputs, outputs=conv_output)
        model.summary()

        if self.verbose:
            print(model)

        self.model = model

        if self.compile_model:
            self._compile_model()
