import numpy as np
import tensorflow as tf
import keras.models
from keras.models import Sequential
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Activation
from keras.layers.merge import add as keras_add
from subtle.dnn.generators.base import GeneratorBase

class GeneratorMultiRes2D(GeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_model()

    def _conv_bn(self, input, params, name):
        params['name'] = name
        x = Conv2D(**params)(input)

        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def _res_block(self, num_channels, input, prefix, res_idx, alpha=1.67, cfracs=[0.1667, 0.3333, 0.5]):
        # when alpha = 1 and cfracs = [1, 1, 1] mres with full blown params is trained
        nc = alpha * num_channels

        conv_params = {
            'kernel_size': (3, 3),
            'padding': 'same',
            'activation': 'relu'
        }

        shortcut = input
        c = [int(nc * cf) for cf in cfracs]
        ct = np.sum(c)

        name_append = '{}_{}'.format(prefix, res_idx)

        conv_params['kernel_size'] = (1, 1)
        conv_params['filters'] = ct
        shortcut = self._conv_bn(
            shortcut,
            conv_params,
            name='conv_rblock_{}_s'.format(name_append)
        )

        conv_params['kernel_size'] = (3, 3)

        conv_params['filters'] = c[0]
        conv_a = self._conv_bn(
            input,
            conv_params,
            name='conv_rblock_{}_a'.format(name_append)
        )

        conv_params['filters'] = c[1]
        conv_b = self._conv_bn(
            input,
            conv_params,
            name='conv_rblock_{}_b'.format(name_append)
        )

        conv_params['filters'] = c[2]
        conv_c = self._conv_bn(
            input,
            conv_params,
            name='conv_rblock_{}_c'.format(name_append)
        )

        out = concatenate(
            [conv_a, conv_b, conv_c],
            name='cat_abc_{}'.format(name_append)
        )

        if self.batch_norm:
            out = BatchNormalization()(out)

        out = keras_add(
            [shortcut, out],
            name='cat_s_abc_{}'.format(name_append)
        )
        out = Activation('relu', name='relu_rblock_{}'.format(name_append))(out)

        if self.batch_norm:
            out = BatchNormalization()(out)

        return out


    def _res_path(self, num_channels, length, input, res_idx):
        shortcut = input

        conv_params = {
            'kernel_size': (1, 1),
            'padding': 'same',
            'activation': None
        }

        conv_params['filters'] = num_channels
        shortcut = self._conv_bn(
            shortcut,
            conv_params,
            name='conv_rpath_s_{}'.format(res_idx)
        )

        conv_params['kernel_size'] = (3, 3)
        conv_params['activation'] = 'relu'
        out = self._conv_bn(
            input,
            conv_params,
            name='conv_rpath_out_{}'.format(res_idx)
        )

        out = keras_add(
            [shortcut, out],
            name='add_s_out_{}'.format(res_idx)
        )
        out = Activation('relu', name='relu_rpath_out_{}'.format(res_idx))(out)

        if self.batch_norm:
            out = BatchNormalization()(out)

        for i in np.arange(length - 1):
            shortcut = out

            conv_params['kernel_size'] = (1, 1)
            conv_params['activation'] = None
            shortcut = self._conv_bn(
                shortcut,
                conv_params,
                name='conv_rpath_{}_{}_a'.format(res_idx, i)
            )

            conv_params['kernel_size'] = (3, 3)
            conv_params['activation'] = 'relu'
            out = self._conv_bn(
                out,
                conv_params,
                name='conv_rpath_{}_{}_b'.format(res_idx, i)
            )

            out = keras_add(
                [shortcut, out],
                name='add_rpath_{}_{}_ab'.format(res_idx, i)
            )
            out = Activation(
                'relu',
                name='relu_rpath_{}_{}'.format(res_idx, i)
            )(out)

            if self.batch_norm:
                out = BatchNormalization()(out)

        return out

    def _build_model(self):
        print('Building respath model...')

        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input), name='model_input')

        nc = self.num_channel_first

        # encoder
        rb1 = self._res_block(nc, inputs, 'enc', 0)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='maxpool_0')(rb1)
        rb1 = self._res_path(nc, 4, rb1, 0)

        rb2 = self._res_block(nc * 2, pool1, 'enc', 1)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(rb2)
        rb2 = self._res_path(nc * 2, 4, rb2, 1)

        rb3 = self._res_block(nc * 4, pool2, 'enc', 2)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(rb3)
        rb3 = self._res_path(nc * 4, 4, rb3, 2)

        # bottleneck
        rb4 = self._res_block(nc * 4, pool3, 'center', 3)

        # decoder
        upsample = lambda idx: UpSampling2D(size=(2, 2), name='upsample_{}'.format(idx))

        up3 = concatenate(
            [upsample(0)(rb4), rb3],
            name='cat_dec_0'
        )
        rb5 = self._res_block(nc * 4, up3, 'dec', 0)

        up4 = concatenate(
            [upsample(1)(rb5), rb2],
            name='cat_dec_1'
        )
        rb6 = self._res_block(nc * 2, up4, 'dec', 1)

        up5 = concatenate(
            [upsample(2)(rb6), rb1],
            name='cat_dec_2'
        )
        rb7 = self._res_block(nc, up5, 'dec', 2)

        # model output
        conv_output = Conv2D(
            self.num_channel_output,
            kernel_size=(1, 1),
            padding='same',
            activation=self.final_activation,
            name='model_output'
        )(rb7)

        if self.verbose:
            print(conv_output)

        # model
        model = keras.models.Model(inputs=inputs, outputs=conv_output)

        if self.verbose:
            print(model)

        model.summary()
        self.model = model

        if self.compile_model:
            self._compile_model()
