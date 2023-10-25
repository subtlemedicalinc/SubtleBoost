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
    def __init__(self, **kwargs):
        self.model_name = 'wdsr3d'
        super().__init__(**kwargs)
        self.features = self.num_filters_first_conv

        self._build_model()

    def _resblock(self, x_in, res_idx):
        x = x_in

        for i in range(self.num_res_convs):
            cname = 'conv_{}_{}'.format(res_idx, i)

            if i == 0:
                filters = self.features * self.res_expansion
            elif i == 1:
                filters = self.features if self.model_config == 'base' else int(self.features * self.linear_factor)
            elif i == 2:
                filters = self.features

            x = Conv3D(
                filters,
                kernel_size=self.get_config('kernel_size', cname),
                padding=self.get_config('padding', cname),
                activation=self.get_config('activation', cname),
                name=cname
            )(x)

        x = keras_add([x_in, x], name='add_{}'.format(res_idx))

        if self.scale_factor:
            sf = self.scale_factor
            x = Lambda(lambda t: t * sf, name='lambda_{}'.format(res_idx))(x)
        return x

    def _build_model(self):
        print('Building WDSR {} {} 3D model...'.format(self.model_name, self.model_config))

        inputs = Input(shape=(self.img_rows, self.img_cols, self.img_depth, self.num_channel_input), name='model_input')

        if self.verbose:
            print(inputs)

        x = inputs
        if self.verbose:
            print(x)

        conv = Conv3D(
            self.features,
            kernel_size=self.get_config('kernel_size', 'conv_init'),
            padding=self.get_config('padding', 'conv_init'),
            name='conv_init'
        )(x)
        if self.verbose:
            print(conv)

        for idx in range(self.num_residuals):
            conv = self._resblock(conv, res_idx=idx)

        conv = Conv3D(
            self.scale ** 3,
            kernel_size=self.get_config('kernel_size', 'conv_a'),
            strides=self.get_config('strides', 'conv_a'),
            padding=self.get_config('padding', 'conv_a'),
            name='conv_a'
        )(conv)
        conv = Conv3DTranspose(
            self.scale ** 3,
            kernel_size=self.get_config('kernel_size', 'conv_trans_a'),
            strides=self.get_config('strides', 'conv_trans_a'),
            padding=self.get_config('padding', 'conv_trans_a'),
            name='conv_trans_a'
        )(conv)

        if self.verbose:
            print('conv', conv)

        conv_1 = Conv3D(
            self.scale ** 3,
            kernel_size=self.get_config('kernel_size', 'conv_b'),
            padding=self.get_config('padding', 'conv_b'),
            name='conv_b'
        )(x)
        conv_1 = MaxPooling3D(
            pool_size=self.get_config('pool_size', 'maxpool_b'),
            name='maxpool_b'
        )(conv_1)
        conv_1 = Conv3DTranspose(
            self.scale ** 3,
            kernel_size=self.get_config('kernel_size', 'conv_trans_b'),
            strides=self.get_config('strides', 'conv_trans_b'),
            padding=self.get_config('padding', 'conv_trans_b'),
            name='conv_trans_b'
        )(conv_1)

        if self.verbose:
            print('conv 1', conv_1)

        conv_output = keras_add([conv, conv_1], name='add_ab')
        conv_output = Conv3D(
            self.num_channel_output,
            kernel_size=self.get_config('kernel_size', 'model_output'),
            padding=self.get_config('padding', 'model_output'),
            name='model_output'
        )(conv_output)

        if self.verbose:
            print('output', conv_output)

        model = keras.models.Model(inputs=inputs, outputs=conv_output)
        model.summary()

        if self.verbose:
            print(model)

        self.model = model

        if self.compile_model:
            self._compile_model()
