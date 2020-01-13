import numpy as np
import keras
from keras.layers import LeakyReLU, Input, Activation, Conv2D, Flatten, Dense
from keras.optimizers import Adam

from subtle.dnn.adversaries.base import AdversaryBase
from subtle.subtle_loss import wasserstein_loss

class AdversaryDense2D(AdversaryBase):
    def __init__(self, **kwargs):
        self.model_name = 'dense2d'

        super().__init__(**kwargs)
        self._build_model()

    def get_real_gt(self, size):
        return np.ones((size, 1))

    def get_fake_gt(self, size):
        return np.zeros((size, 1))

    def _conv_block(self, input, filters, idx, suffix=''):
        cname = 'conv_{}_{}'.format(idx, suffix)
        d_out = Conv2D(
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

        return d_out

    def _build_model(self):
        input = Input(
            shape=(self.img_rows, self.img_cols, self.num_channel_input), name='adv_input'
        )

        d_out = input

        for idx in range(self.num_convs):
            d_out = self._conv_block(d_out, self.num_filters_first_conv, idx=idx, suffix='a')

            if self.verbose:
                print(d_out)

        d_out = Flatten()(d_out)
        d_out = Dense(self.dense_filters, name='final_dense')(d_out)

        dense_act = self.get_config('activation', 'final_dense')
        if dense_act == 'relu':
            d_out = Activation(dense_act)(d_out)
        else:
            d_out = LeakyReLU(alpha=self.get_config('lrelu_alpha', 'final_dense'))(d_out)

        if self.verbose:
            print(d_out)

        d_out = Dense(
            self.num_channel_output,
            activation=self.get_config('activation', 'adv_output'),
            name='adv_output'
        )(d_out)

        model = keras.models.Model(inputs=input, outputs=d_out)
        model.summary()
        self.model = model

        if self.compile_model:
            self._compile_model()
