import tensorflow as tf
import keras.models
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Activation, ReLU, LeakyReLU
from keras.layers.merge import add as keras_add

from subtle.dnn.generators.base import GeneratorBase

class GeneratorUNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_model()

    def _conv(self, x, filters, kernel_size=(3, 3), padding='same', activation='relu'):
        out = Conv2D(
            filters=filters, kernel_size=kernel_size, padding=padding
        )(x)
        act_fn = LeakyReLU(alpha=0.2) if activation == 'leaky_relu' else ReLU()
        return act_fn(out)

    def _build_model(self):
        print('Building standard model...')
        # batch norm
        if self.batch_norm:
            lambda_bn = lambda x: BatchNormalization()(x)
        else:
            lambda_bn = lambda x: x

        # layers
        # 2D input is (rows, cols, channels)

        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input))

        if self.verbose:
            print(inputs)

        # step 1
        conv1 = inputs

        for i in range(self.num_conv_per_pooling):
            conv1 = self._conv(conv1, filters=self.num_channel_first)
            conv1 = lambda_bn(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        if self.verbose:
            print(conv1, pool1)

        # encoder pools
        convs = [inputs, conv1]
        pools = [inputs, pool1]
        list_num_features = [self.num_channel_input, self.num_channel_first]

        for i in range(1, self.num_poolings):

            # step 2
            conv_encoder = pools[-1]
            num_channel = self.num_channel_first * (2**i) # double channels each step
            # FIXME: check if this should be 2**i ?

            for j in range(self.num_conv_per_pooling):

                conv_encoder = self._conv(conv_encoder, filters=num_channel)
                conv_encoder = lambda_bn(conv_encoder)

            pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)

            if self.verbose:
                print(conv_encoder, pool_encoder)

            pools.append(pool_encoder)
            convs.append(conv_encoder)
            list_num_features.append(num_channel)

        # center connection
        conv_center = self._conv(pools[-1], filters=list_num_features[-1])

        print('conv center before add', conv_center)
        # residual connection
        conv_center = keras_add([pools[-1], conv_center])

        if self.verbose:
            print('conv center...', conv_center)

        # decoder steps
        conv_decoders = [conv_center]

        for i in range(1, self.num_poolings + 1):
            decoder_upsample = UpSampling2D(size=(2, 2))(conv_decoders[-1])
            up_decoder = concatenate([decoder_upsample, convs[-i]])
            conv_decoder = up_decoder

            for j in range(self.num_conv_per_pooling):
                conv_decoder = self._conv(
                    conv_decoder,
                    filters=list_num_features[-i]
                )
                conv_decoder = lambda_bn(conv_decoder)

            conv_decoders.append(conv_decoder)

            if self.verbose:
                    print(conv_decoder, up_decoder)

        # output layer

        conv_decoder = conv_decoders[-1]

        conv_output = self._conv(
            conv_decoder, filters=self.num_channel_output,
            kernel_size=(1, 1), activation=self.final_activation
        )

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
