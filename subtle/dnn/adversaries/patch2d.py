import keras
from keras.layers import LeakyReLU, BatchNormalization, Input, Activation
from keras.optimizers import Adam

from subtle.dnn.layers.SpectralNormalization import ConvSN2D

class AdversaryPatch2D:
    def __init__(
        self, num_channel_input=1, img_rows=128, img_cols=128, num_channel_first=32, num_poolings=3,
        batch_norm=True, verbose=True, compile_model=True
    ):
        self.num_channel_input = num_channel_input
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_channel_first = num_channel_first
        self.num_poolings = num_poolings
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.compile_model = True

        self.model = None
        self._build_adversary()

    def _conv_block(self, input, filters, strides=1, bnorm=True):
        d_out = ConvSN2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')(input)
        d_out = LeakyReLU(alpha=0.2)(d_out)

        if bnorm:
            d_out = BatchNormalization(momentum=0.8)(d_out)

        return d_out

    def _build_adversary(self):

        input = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input))

        nc = self.num_channel_first

        d_out = self._conv_block(input, nc, bnorm=False)
        d_out = self._conv_block(d_out, nc, strides=2)
        if self.verbose:
            print(d_out)

        d_out = self._conv_block(d_out, nc * 2)
        d_out = self._conv_block(d_out, nc * 2, strides=2)
        if self.verbose:
            print(d_out)

        d_out = self._conv_block(d_out, nc * 4)
        d_out = self._conv_block(d_out, nc * 4, strides=2)
        if self.verbose:
            print(d_out)

        d_out = self._conv_block(d_out, nc * 8)
        d_out = self._conv_block(d_out, nc * 8, strides=1)
        if self.verbose:
            print(d_out)

        val = ConvSN2D(filters=1, kernel_size=(3, 3), strides=1, padding='same')(d_out)
        val = Activation('sigmoid')(val)

        if self.verbose:
            print(val)

        model = keras.models.Model(inputs=input, outputs=val)
        model.summary()
        self.model = model

        if self.compile_model:
            self._compile_model()

    def _compile_model(self):
        self.model.compile(
            loss='mse', optimizer=Adam(2e-3, 0.5)
        )
