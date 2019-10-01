"""
Based on WDSR implementation in https://github.com/krasserm/super-resolution
"""

import tensorflow as tf
from keras import backend as K
import keras.models
from keras.layers import Input, Conv3D, Activation, Lambda, MaxPooling3D, Conv3DTranspose
from keras.layers.merge import add as keras_add
from keras.initializers import he_normal

from subtle.dnn.generators.base import GeneratorBase

class GeneratorWDSR3D(GeneratorBase):
    def __init__(self, num_layers=6, scale_factor=0.1, scale=2, wdsr_type='wdsr_a', **kwargs):
        super().__init__(**kwargs)
        self.wdsr_type = wdsr_type
        self.scale_factor = scale_factor
        self.scale = 2
        self.features = self.num_channel_first
        self.num_layers = num_layers

        self.param_map = {
            'wdsr_a': {
                'res_expansion': 4,
                'res_block': self._resblock_a
            },
            'wdsr_b': {
                'res_expansion': 6,
                'res_block': self._resblock_b
            }
        }

        self._build_model()

    def _get_params(self):
        return self.param_map[self.wdsr_type]

    def _resblock_a(self, x_in):
        params = self._get_params()
        x = Conv3D(self.features * params['res_expansion'], kernel_size=3, padding='same')(x_in)
        x = Activation('relu')(x)
        x = Conv3D(self.features, kernel_size=3, padding='same')(x)
        x = keras_add([x_in, x])

        if self.scale_factor:
            x = Lambda(lambda t: t * self.scale_factor)(x)
        return x

    def _resblock_b(self, x_in):
        linear = 0.8
        params = self._get_params()
        x = Conv3D(self.features * params['res_expansion'], kernel_size=1, padding='same')(x_in)
        x = Activation('relu')(x)
        x = Conv3D(int(self.features * linear), kernel_size=1, padding='same')(x)
        x = Conv3D(self.features, kernel_size=3, padding='same')(x)
        x = keras_add([x_in, x])

        if self.scale_factor:
            x = Lambda(lambda t: t * self.scale_factor)(x)
        return x

    def _build_model(self):
        print('Building WDSR {} 3D model...'.format(self.wdsr_type))
        params = self._get_params()

        inputs = Input(shape=(self.img_rows, self.img_cols, self.img_depth, self.num_channel_input))

        if self.verbose:
            print(inputs)

        x = inputs
        if self.verbose:
            print(x)

        conv = Conv3D(self.features, kernel_size=3, padding='valid')(x)
        if self.verbose:
            print(conv)

        for _ in range(self.num_layers):
            conv = params['res_block'](conv)

        conv = Conv3D(self.scale ** 3, kernel_size=3, strides=(2, 2, 2), padding='valid')(conv)
        conv = Conv3DTranspose(self.scale ** 3, kernel_size=3, strides=(2, 2, 2), padding='same')(conv)

        if self.verbose:
            print('conv', conv)

        conv_1 = Conv3D(self.scale ** 3, kernel_size=5, padding='valid')(x)
        conv_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_1)
        conv_1 = Conv3DTranspose(self.scale ** 3, kernel_size=5, strides=(2, 2, 2), padding='same')(conv_1)

        if self.verbose:
            print('conv 1', conv_1)

        conv_output = keras_add([conv, conv_1])
        conv_output = self._spatial_padding()(conv_output)
        conv_output = Conv3D(1, kernel_size=3, padding='same')(conv_output)

        if self.verbose:
            print('output', conv_output)

        model = keras.models.Model(inputs=inputs, outputs=conv_output)
        model.summary()

        if self.verbose:
            print(model)

        self.model = model

        if self.compile_model:
            self._compile_model()

    def _spatial_padding(self):
        return Lambda(
            lambda x: K.spatial_3d_padding(x, padding=((2, 2), (2, 2), (2, 2)))
        )
