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

    def _res_block(self, num_channels, input, prefix, res_idx, alpha=1, cfracs=[0.1667, 0.3333, 0.5]):
        # when alpha = 1.67 and cfracs = [0.1667, 0.3333, 0.5] the architecture is similar to the one proposed in https://arxiv.org/pdf/1902.04049.pdf
        nc = alpha * num_channels

        conv_params = {
            'kernel_size': (3, 3),
            'padding': 'same',
            'activation': 'relu'
        }

        shortcut = input
        c = [int(nc * cf) for cf in cfracs]
        if self.num_conv_per_pooling > 3:
            c = c + ([1] * self.num_conv_per_pooling - 3)
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
        convs = [input]

        for i in range(self.num_conv_per_pooling):
            conv_params['filters'] = c[i]
            conv_a = self._conv_bn(
                convs[-1],
                conv_params,
                name='conv_rblock_{}_{}'.format(name_append, i)
            )
            convs.append(conv_a)

        out = concatenate(
            convs[1:],
            name='cat_convs_{}'.format(name_append)
        )

        if self.batch_norm:
            out = BatchNormalization()(out)

        out = keras_add(
            [shortcut, out],
            name='cat_s_convs_{}'.format(name_append)
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

        nc = self.num_filters_first_conv

        # encoder
        enc_pool_ip = inputs
        enc_pools = []
        for i in range(self.num_poolings):
            rb = self._res_block(nc * (2 ** i), enc_pool_ip, 'enc', i)
            enc_pool_ip = MaxPooling2D(pool_size=(2, 2), name='maxpool_{}'.format(i))(rb)
            rb = self._res_path(nc * (2 ** i), 4, rb, i)
            enc_pools.append(rb)

            if self.verbose:
                print(rb)

        # bottleneck
        bneck_nc = nc * (2 ** (self.num_poolings - 1))
        center_conv = self._res_block(bneck_nc, enc_pool_ip, 'center', 4)

        if self.verbose:
            print(center_conv)

        # decoder
        upsample = lambda idx: UpSampling2D(size=(2, 2), name='upsample_{}'.format(idx))

        usamples = [center_conv]
        for i in range(self.num_poolings):
            up = concatenate(
                [upsample(i)(usamples[-1]), enc_pools[::-1][i]],
                name='cat_dec_{}'.format(i)
            )
            dec_nc = nc * (2 ** ((self.num_poolings - i) - 1))
            usamples.append(self._res_block(dec_nc, up, 'dec', i))

            if self.verbose:
                print(usamples[-1])

        # model output
        conv_output = Conv2D(
            self.num_channel_output,
            kernel_size=(1, 1),
            padding='same',
            activation=self.final_activation,
            name='model_output'
        )(usamples[-1])

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
