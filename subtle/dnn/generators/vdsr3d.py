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
    def __init__(self, **kwargs):
        self.model_name = 'vdsr3d'
        super().__init__(**kwargs)
        
        self._build_model()

    def _conv(self, x, features=None, name=None):
        features = self.num_filters_first_conv if features is None else features

        init_conf = self.get_config('kernel_initializer')
        if init_conf == 'he_normal':
            kernel_init = he_normal(seed=self.init_seed)
        else:
            kernel_init = init_conf

        return Conv3D(
            features,
            kernel_size=self.get_config('kernel_size'),
            strides=1,
            activation=self.get_config('activation'),
            kernel_initializer=kernel_init,
            padding=self.get_config('padding'),
            name=name
        )(x)

    def _build_model(self):
        print('Building VDSR 3D model...')
        inputs = Input(shape=(self.img_rows, self.img_cols, self.img_depth, self.num_channel_input), name='model_input')

        if self.verbose:
            print(inputs)

        conv = inputs
        for res_idx in range(self.num_residuals-1):
            cname = 'conv_{}'.format(res_idx)
            conv = self._conv(conv, name=cname)

        pre_act = Activation(
            self.get_config('activation', 'act_conv_pre_out'),
            name='act_conv_pre_out'
        )
        pre_conv = self._conv(
            conv,
            features=self.num_channel_output,
            name='conv_pre_out'
        )
        conv = pre_act(pre_conv)

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
