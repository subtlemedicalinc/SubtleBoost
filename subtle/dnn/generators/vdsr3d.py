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

    def _conv(self, x, features=None, name=None):
        features = self.num_channel_first if features is None else features
        return Conv3D(
            features,
            kernel_size=3,
            strides=1,
            activation=None,
            kernel_initializer=he_normal(seed=self.init_seed),
            padding='same',
            name=name
        )(x)

    def _build_model(self):
        print('Building VDSR 3D model...')
        inputs = Input(shape=(self.img_rows, self.img_cols, self.img_depth, self.num_channel_input), name='model_input')

        if self.verbose:
            print(inputs)

        conv = inputs
        for res_idx in range(self.num_layers-1):
            cname = 'conv_{}'.format(res_idx)

            conv = Activation(
                'relu',
                name='relu_{}'.format(cname)
            )(self._conv(conv, name=cname))

        conv = Activation(
            'tanh',
            name='tanh_conv_pre_out'
        )(self._conv(conv, features=1, name='conv_pre_out'))
        conv_output = keras_add([inputs, conv], name='add_pre_out')
        conv_output = self._conv(
            conv_output,
            features=self.num_channel_output,
            name='model_output'
        )

        if self.verbose:
            print('final output', conv_output)
        # model
        model = keras.models.Model(inputs=inputs, outputs=conv_output)
        model.summary()

        if self.verbose:
            print(model)

        self.model = model

        if self.compile_model:
            self._compile_model()
