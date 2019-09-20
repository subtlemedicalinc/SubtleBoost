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

    def _conv_bn(self, input, params):
        x = Conv2D(**params)(input)

        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def _res_block(self, num_channels, input, alpha=1.67):
        #nc = alpha * num_channels
        nc = num_channels

        conv_params = {
            'kernel_size': (3, 3),
            'padding': 'same',
            'activation': 'relu'
        }
        shortcut = input

        # c1 = int(nc * (1/6))
        # c2 = int(nc * (1/3))
        # c3 = int(nc * (1/2))
        c1 = c2 = c3 = nc

        ct = (c1 + c2 + c3)

        conv_params['kernel_size'] = (1, 1)
        conv_params['filters'] = ct
        shortcut = self._conv_bn(shortcut, conv_params)

        conv_params['kernel_size'] = (3, 3)

        conv_params['filters'] = c1
        conv_a = self._conv_bn(input, conv_params)

        conv_params['filters'] = c2
        conv_b = self._conv_bn(input, conv_params)

        conv_params['filters'] = c3
        conv_c = self._conv_bn(input, conv_params)

        out = concatenate([conv_a, conv_b, conv_c])

        if self.batch_norm:
            out = BatchNormalization()(out)

        out = keras_add([shortcut, out])
        out = Activation('relu')(out)

        if self.batch_norm:
            out = BatchNormalization()(out)

        return out


    def _res_path(self, num_channels, length, input):
        shortcut = input

        conv_params = {
            'kernel_size': (1, 1),
            'padding': 'same',
            'activation': None
        }

        conv_params['filters'] = num_channels
        shortcut = self._conv_bn(shortcut, conv_params)

        conv_params['kernel_size'] = (3, 3)
        conv_params['activation'] = 'relu'
        out = self._conv_bn(input, conv_params)

        out = keras_add([shortcut, out])
        out = Activation('relu')(out)

        if self.batch_norm:
            out = BatchNormalization()(out)

        for i in np.arange(length - 1):
            shortcut = out

            conv_params['kernel_size'] = (1, 1)
            conv_params['activation'] = None
            shortcut = self._conv_bn(shortcut, conv_params)

            conv_params['kernel_size'] = (3, 3)
            conv_params['activation'] = 'relu'
            out = self._conv_bn(out, conv_params)

            out = keras_add([shortcut, out])
            out = Activation('relu')(out)

            if self.batch_norm:
                out = BatchNormalization()(out)

        return out

    def _build_model(self):
        print('Building respath model...')

        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input))

        nc = self.num_channel_first

        # encoder
        rb1 = self._res_block(nc, inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(rb1)
        rb1 = self._res_path(nc, 4, rb1)

        rb2 = self._res_block(nc * 2, pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(rb2)
        rb2 = self._res_path(nc * 2, 4, rb2)

        rb3 = self._res_block(nc * 4, pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(rb3)
        rb3 = self._res_path(nc * 4, 4, rb3)

        # bottleneck
        rb4 = self._res_block(nc * 4, pool3)

        # decoder
        upsample = UpSampling2D(size=(2, 2))

        up3 = concatenate([upsample(rb4), rb3])
        rb5 = self._res_block(nc * 4, up3)

        up4 = concatenate([upsample(rb5), rb2])
        rb6 = self._res_block(nc * 2, up4)

        up5 = concatenate([upsample(rb6), rb1])
        rb7 = self._res_block(nc, up5)

        # model output
        conv_output = Conv2D(self.num_channel_output, (1, 1), padding="same", activation=self.final_activation)(rb7)

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
