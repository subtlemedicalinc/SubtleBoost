import numpy as np
import keras
from keras.layers import LeakyReLU, BatchNormalization, Input, Activation
from keras.optimizers import Adam

from subtle.dnn.adversaries.base import AdversaryBase
from subtle.dnn.layers.SpectralNormalization import ConvSN2D

class AdversaryPatch2D(AdversaryBase):
    def __init__(self, **kwargs):
        self.model_name = 'patch2d'
        self.patch_size = 15

        super().__init__(**kwargs)
        self._build_model()

    def get_real_gt(self, size):
        return np.ones((size, self.patch_size, self.patch_size, 1))

    def get_fake_gt(self, size):
        return np.zeros((size, self.patch_size, self.patch_size, 1))

    def _conv_block(self, input, filters, idx, suffix=''):
        cname = 'conv_{}_{}'.format(idx, suffix)
        d_out = ConvSN2D(
            filters=filters,
            kernel_size=self.get_config('kernel_size', cname),
            strides=self.get_config('strides', cname),
            padding=self.get_config('padding', cname),
            name=cname
        )(input)

        act_name = 'lrelu_conv_{}_{}'.format(idx, suffix)
        d_out = LeakyReLU(
            alpha=self.get_config('lrelu_alpha', act_name), name=act_name
        )(d_out)

        if self.get_config('batch_norm', cname):
            d_out = BatchNormalization(momentum=self.bnorm_momentum)(d_out)

        return d_out

    def _build_model(self):
        input = Input(
            shape=(self.img_rows, self.img_cols, self.num_channel_input), name='adv_input'
        )

        d_out = input

        for idx in range(self.num_convs):
            nc = self.num_filters_first_conv * (2 ** idx)
            d_out = self._conv_block(d_out, nc, idx=idx, suffix='a')
            d_out = self._conv_block(d_out, nc, idx=idx, suffix='b')
            if self.verbose:
                print(d_out)

        fcname = 'final_conv'
        val = ConvSN2D(
            filters=self.num_channel_output,
            kernel_size=self.get_config('kernel_size', fcname),
            strides=self.get_config('strides', fcname),
            padding=self.get_config('padding', fcname),
            name=fcname
        )(d_out)
        val = Activation(
            self.get_config('activation', 'adv_output'),
            name='adv_output'
        )(val)

        if self.verbose:
            print(val)

        model = keras.models.Model(inputs=input, outputs=val)
        model.summary()
        self.model = model

        if self.compile_model:
            self._compile_model()
