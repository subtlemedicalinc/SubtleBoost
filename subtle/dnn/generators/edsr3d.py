"""
Based on EDSR implementation in SubtleMR by Long Wang
"""

import tensorflow as tf
import keras.models
from keras.layers import Input, Conv3D, Activation
from keras.layers.merge import add as keras_add
from keras.initializers import he_normal

from subtle.dnn.generators.base import GeneratorBase
from subtle.dnn.layers.ConstMultiplier import ConstMultiplier


class GeneratorEDSR3D(GeneratorBase):
    def __init__(self, num_residuals=16, scale_factor=0.1, init_seed=15213, **kwargs):
        super().__init__(**kwargs)
        self.init_seed = init_seed
        self.scale_factor = scale_factor
        self.num_residuals = num_residuals

        self._build_model()

    def _conv(self, x, features=None, name=None):
        features = self.num_filters_first_conv if features is None else features
        return Conv3D(
            features, kernel_size=3, strides=1,
            kernel_initializer=he_normal(seed=self.init_seed),
            padding='same', name=name
        )(x)

    def _resblock(self, x, res_idx):
        conv_name = 'conv_res_{}'.format(res_idx)
        res_conv1 = Activation(
            'relu',
            name='relu_{}'.format(conv_name)
        )(self._conv(x, name='{}_1'.format(conv_name)))

        res_conv2 = self._conv(
            res_conv1, name='{}_2'.format(conv_name)
        )
        const_mult = ConstMultiplier(
            val=self.scale_factor,
            name='noise_mul_{}'.format(res_idx)
        )(res_conv2)
        return keras_add([x, const_mult], name='add_res_{}'.format(res_idx))

    def _build_model(self):
        print('Building EDSR 3D model...')
        inputs = Input(shape=(self.img_rows, self.img_cols, self.img_depth, self.num_channel_input), name='model_input')

        if self.verbose:
            print(inputs)

        x = self._conv(inputs, name='conv_init')
        init_conv = x

        if self.verbose:
            print(init_conv)

        for res_idx in range(self.num_residuals):
            x = self._resblock(x, res_idx)

            if self.verbose:
                print(x)

        x = self._conv(x, name='conv_final')
        if self.verbose:
            print(x)

        x = keras_add([x, init_conv], name='add_conv_final')
        if self.verbose:
            print(x)

        conv_output = self._conv(x, features=self.num_channel_output, name='model_output')

        if self.verbose:
            print(conv_output)

        # model
        model = keras.models.Model(inputs=inputs, outputs=conv_output)
        model.summary()

        if self.verbose:
            print(model)

        self.model = model

        if self.compile_model:
            self._compile_model()
