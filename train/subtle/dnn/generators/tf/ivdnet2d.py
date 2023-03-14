"""
Based on PyTorch implementation of IVD-Net (https://arxiv.org/pdf/1811.08305.pdf) from
https://github.com/josedolz/IVD-Net
"""
import tensorflow as tf
import numpy as np
import keras.models
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, BatchNormalization
from keras.layers import MaxPooling2D, concatenate, Average, Lambda
from keras.layers.merge import add as keras_add
from keras.initializers import glorot_normal
import keras.backend as K

from subtle.dnn.generators.base import GeneratorBase
from subtle.dnn.layers.ConstantPadding2D import ConstantPadding2D

class GeneratorIVDNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'ivdnet2d'
        super().__init__(**kwargs)

        self._build_model()

    def _bnorm(self, x):
        if self.batch_norm:
            lambda_bn = lambda x: BatchNormalization()(x)
        else:
            lambda_bn = lambda x: x
        return lambda_bn(x)

    def _conv(
        self, x, filters, kernel_size=3, padding=1, strides=1, dilation_rate=1, activation='relu',
        name=None
    ):
        act_fn = Activation(activation, name='{}_{}'.format(activation, name))

        out = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
        dilation_rate=dilation_rate, name=name, kernel_initializer=glorot_normal())(x)

        out = ConstantPadding2D(padding=(padding, padding), name='{}_cpad'.format(name))(out)

        out = self._bnorm(out)
        return act_fn(out)

    def _conv_asym_inception(
        self, x, filters, kernel_size=3, padding=1, strides=1, dilation_rate=1, activation='relu',
        name=None
    ):

        act_fn1 = Activation(activation, name='{}_{}_1'.format(activation, name))
        act_fn2 = Activation(activation, name='{}_{}_2'.format(activation, name))

        inc1 = Conv2D(
            filters=filters, kernel_size=(kernel_size, 1), dilation_rate=(dilation_rate, 1),
            kernel_initializer=glorot_normal(), name='{}_i1'.format(name)
        )(x)
        inc1 = ConstantPadding2D(padding=(padding, 0), name='{}_cpad1'.format(name))(inc1)
        inc1 = self._bnorm(inc1)
        inc1 = act_fn1(inc1)

        inc2 = Conv2D(
            filters=filters, kernel_size=(1, kernel_size), dilation_rate=(1, dilation_rate),
            kernel_initializer=glorot_normal(), name='{}_i2'.format(name)
        )(inc1)
        inc2 = ConstantPadding2D(padding=(0, padding), name='{}_cpad2'.format(name))(inc2)
        inc2 = self._bnorm(inc2)
        inc2 = act_fn2(inc2)
        return inc2

    def _deconv(self, x, filters, activation='relu', name=None):
        out = Conv2DTranspose(
            filters=filters, kernel_size=3, strides=2, name=name, padding="same"
        )(x)
        out = self._bnorm(out)
        return Activation(activation=activation, name='{}_{}'.format(activation, name))(out)

    def _maxpool(self, x, name=None):
        return MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name=name)(x)

    def _resconv_inception(self, x, filters, asymmetric=False, activation='relu', name=None):
        conv_1 = self._conv(x, filters, name='{}_c1'.format(name))
        incept_block = self._conv_asym_inception if asymmetric else self._conv

        conv_2_1 = incept_block(
            conv_1, filters, activation=activation, kernel_size=1, strides=1, padding=0,
            dilation_rate=1, name='{}_c2_1'.format(name)
        )
        conv_2_2 = incept_block(
            conv_1, filters, activation=activation, kernel_size=3, strides=1, padding=1,
            dilation_rate=1, name='{}_c2_2'.format(name)
        )
        conv_2_3 = incept_block(
            conv_1, filters, activation=activation, kernel_size=5, strides=1, padding=2,
            dilation_rate=1, name='{}_c2_3'.format(name)
        )
        conv_2_4 = incept_block(
            conv_1, filters, activation=activation, kernel_size=3, strides=1, padding=2,
            dilation_rate=2, name='{}_c2_4'.format(name)
        )
        conv_2_5 = incept_block(
            conv_1, filters, activation=activation, kernel_size=3, strides=1, padding=4,
            dilation_rate=4, name='{}_c2_5'.format(name)
        )

        out = concatenate(
            [conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5],
            name='cat_{}'.format(name)
        )
        out = self._conv(
            out, filters, activation=activation, kernel_size=1, strides=1, padding=0, dilation_rate=1, name='{}_incpt_out'.format(name)
        )
        out = keras_add([out, conv_1], name='{}_add'.format(name))
        out = self._conv(out, filters, activation=activation, name='{}_out'.format(name))
        return out

    def _build_model(self):
        print('Building {}-{} model...'.format(self.model_name, self.model_config))

        # layers
        # 2D input is (rows, cols, channels)

        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input), name='model_input')

        print('inputs', inputs)

        num_ips = 4 #t1_pre, t1_low, t2, flair and UAD mask
        ip_names = ['t1_pre', 't1_low', 't2', 'fl']
        t1_pre, t1_low, t2, flair = [
            Lambda(lambda ip: ip[..., idx::num_ips], name=ip_names[idx])(inputs)
            for idx in np.arange(num_ips)
        ]

        nfilter = self.num_filters_first_conv

        ### First Level ###
        down_1_0 = self._resconv_inception(t1_pre, nfilter, asymmetric=True, name='down_1_0')
        down_1_1 = self._resconv_inception(t1_low, nfilter, asymmetric=True, name='down_1_1')
        down_1_2 = self._resconv_inception(t2, nfilter, asymmetric=True, name='down_1_2')
        down_1_3 = self._resconv_inception(flair, nfilter, asymmetric=True, name='down_1_3')

        ### Second Level ###
        input_2_0 = concatenate([
            self._maxpool(down_1_0), self._maxpool(down_1_1), self._maxpool(down_1_2),
            self._maxpool(down_1_3)
        ])

        input_2_1 = concatenate([
            self._maxpool(down_1_1), self._maxpool(down_1_2), self._maxpool(down_1_3),
            self._maxpool(down_1_0)
        ])

        input_2_2 = concatenate([
            self._maxpool(down_1_2), self._maxpool(down_1_3), self._maxpool(down_1_0),
            self._maxpool(down_1_1)
        ])

        input_2_3 = concatenate([
            self._maxpool(down_1_3), self._maxpool(down_1_0), self._maxpool(down_1_1),
            self._maxpool(down_1_2)
        ])

        down_2_0 = self._resconv_inception(input_2_0, nfilter*2, asymmetric=True, name='down_2_0')
        down_2_1 = self._resconv_inception(input_2_1, nfilter*2, asymmetric=True, name='down_2_1')
        down_2_2 = self._resconv_inception(input_2_2, nfilter*2, asymmetric=True, name='down_2_2')
        down_2_3 = self._resconv_inception(input_2_3, nfilter*2, asymmetric=True, name='down_2_3')

        ### Third Level ###

        down_2_0m = self._maxpool(down_2_0)
        down_2_1m = self._maxpool(down_2_1)
        down_2_2m = self._maxpool(down_2_2)
        down_2_3m = self._maxpool(down_2_3)

        crop_lambda = Lambda(lambda img: tf.image.central_crop(img, 0.5))

        input_3_0 = concatenate([down_2_0m, down_2_1m, down_2_2m, down_2_3m])
        input_3_0 = concatenate([input_3_0, crop_lambda(input_2_0)])

        input_3_1 = concatenate([down_2_1m, down_2_2m, down_2_3m, down_2_0m])
        input_3_1 = concatenate([input_3_1, crop_lambda(input_2_1)])

        input_3_2 = concatenate([down_2_2m, down_2_3m, down_2_0m, down_2_1m])
        input_3_2 = concatenate([input_3_2, crop_lambda(input_2_2)])

        input_3_3 = concatenate([down_2_3m, down_2_0m, down_2_1m, down_2_2m])
        input_3_3 = concatenate([input_3_3, crop_lambda(input_2_3)])

        down_3_0 = self._resconv_inception(input_3_0, nfilter*4, asymmetric=True, name='down_3_0')
        down_3_1 = self._resconv_inception(input_3_1, nfilter*4, asymmetric=True, name='down_3_1')
        down_3_2 = self._resconv_inception(input_3_2, nfilter*4, asymmetric=True, name='down_3_2')
        down_3_3 = self._resconv_inception(input_3_3, nfilter*4, asymmetric=True, name='down_3_3')

        ### Fourth Level ###
        down_3_0m = self._maxpool(down_3_0)
        down_3_1m = self._maxpool(down_3_1)
        down_3_2m = self._maxpool(down_3_2)
        down_3_3m = self._maxpool(down_3_3)

        input_4_0 = concatenate([down_3_0m, down_3_1m, down_3_2m, down_3_3m])
        input_4_0 = concatenate([input_4_0, crop_lambda(input_3_0)])

        input_4_1 = concatenate([down_3_1m, down_3_2m, down_3_3m, down_3_0m])
        input_4_1 = concatenate([input_4_1, crop_lambda(input_3_1)])

        input_4_2 = concatenate([down_3_2m, down_3_3m, down_3_0m, down_3_1m])
        input_4_2 = concatenate([input_4_2, crop_lambda(input_3_2)])

        input_4_3 = concatenate([down_3_3m, down_3_0m, down_3_1m, down_3_2m])
        input_4_3 = concatenate([input_4_3, crop_lambda(input_3_3)])

        down_4_0 = self._resconv_inception(input_4_0, nfilter*8, asymmetric=True, name='down_4_0')
        down_4_1 = self._resconv_inception(input_4_1, nfilter*8, asymmetric=True, name='down_4_1')
        down_4_2 = self._resconv_inception(input_4_2, nfilter*8, asymmetric=True, name='down_4_2')
        down_4_3 = self._resconv_inception(input_4_3, nfilter*8, asymmetric=True, name='down_4_3')

        ####--- Bridge---####
        # Max-pool
        down_4_0m = self._maxpool(down_4_0)
        down_4_1m = self._maxpool(down_4_1)
        down_4_2m = self._maxpool(down_4_2)
        down_4_3m = self._maxpool(down_4_3)

        input_bridge = concatenate([down_4_0m, down_4_1m, down_4_2m, down_4_3m], name='ip_bridge')
        if self.img_rows == 240:
            resize_lambda = Lambda(lambda img: tf.image.resize_image_with_pad(img, 15, 15))

            input_bridge = resize_lambda(input_bridge)
            input_4_0_crop = crop_lambda(input_4_0)
            input_4_0_crop = resize_lambda(input_4_0_crop)

        input_bridge = concatenate([input_bridge, input_4_0_crop])
        bridge = self._resconv_inception(input_bridge, nfilter*16, asymmetric=True, name='bridge')

        ####--- Decoder ---####
        deconv_1 = self._deconv(bridge, nfilter*8, name='deconv_1')
        # residual connection
        skip_1 = Average(name='skip_1')([deconv_1, down_4_0, down_4_1, down_4_2, down_4_3])
        up_1 = self._resconv_inception(skip_1, nfilter*8, name='up_1')

        deconv_2 = self._deconv(up_1, nfilter*4, name='deconv_2')
        # residual connection
        skip_2 = Average(name='skip_2')([deconv_2, down_3_0, down_3_1, down_3_2, down_3_3])
        up_2 = self._resconv_inception(skip_2, nfilter*4, name='up_2')

        deconv_3 = self._deconv(up_2, nfilter*2, name='deconv_3')
        # residual connection
        skip_3 = Average(name='skip_3')([deconv_3, down_2_0, down_2_1, down_2_2, down_2_3])
        up_3 = self._resconv_inception(skip_3, nfilter*2, name='up_3')

        deconv_4 = self._deconv(up_3, nfilter, name='deconv_4')
        # residual connection
        skip_4 = Average(name='skip_4')([deconv_4, down_1_0, down_1_1, down_1_2, down_1_3])
        up_4 = self._resconv_inception(skip_4, nfilter, name='up_4')

        conv_output = Conv2D(
            filters=self.num_channel_output, kernel_size=3, strides=1, padding='same',
            kernel_initializer=glorot_normal(), name='model_output'
        )(up_4)

        model = keras.models.Model(inputs=inputs, outputs=conv_output)
        model.summary()

        if self.verbose:
            print(model)

        self.model = model

        if self.compile_model:
            self._compile_model()
