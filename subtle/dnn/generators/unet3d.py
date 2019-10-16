import tensorflow as tf
import keras.models
from keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, UpSampling3D, concatenate, Activation, ReLU, LeakyReLU
from keras.layers.merge import add as keras_add

from subtle.dnn.generators.base import GeneratorBase

class GeneratorUNet3D(GeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_model()

    def _conv(self, x, filters, kernel_size=(3, 3), padding='same', activation='relu', name=None):
        out = Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            name=name
        )(x)

        act_fn = (
            LeakyReLU(alpha=0.2, name='lrelu_{}'.format(name))
            if activation == 'leaky_relu' else
            ReLU(name='relu_{}'.format(name))
        )
        return act_fn(out)

    def _build_model(self):
        print('Building standard 3D model...')
        # batch norm
        if self.batch_norm:
            lambda_bn = lambda x: BatchNormalization()(x)
        else:
            lambda_bn = lambda x: x

        inputs = Input(shape=(self.img_rows, self.img_cols, self.img_depth, self.num_channel_input), name="model_input")

        if self.verbose:
            print(inputs)

        # step 1
        conv1 = inputs

        for i in range(self.num_conv_per_pooling):
            conv1 = self._conv(
                conv1,
                filters=self.num_filters_first_conv,
                kernel_size=3,
                padding='same',
                activation='relu',
                name='conv_enc_1_{}'.format(i)
            )
            conv1 = lambda_bn(conv1)

        pool1 = MaxPooling3D(pool_size=(2, 2, 2), name='maxpool_1')(conv1)

        if self.verbose:
            print(conv1, pool1)

        # encoder pools
        convs = [inputs, conv1]
        pools = [inputs, pool1]
        list_num_features = [self.num_channel_input, self.num_filters_first_conv]

        for i in range(1, self.num_poolings):

            # step 2
            conv_encoder = pools[-1]
            num_channel = self.num_filters_first_conv * (2**i) # double channels each step
            # FIXME: check if this should be 2**i ?

            for j in range(self.num_conv_per_pooling):

                conv_encoder = self._conv(
                    conv_encoder,
                    filters=num_channel,
                    kernel_size=3,
                    padding='same',
                    activation='relu',
                    name='conv_enc_{}_{}'.format(i + 1, j)
                )
                conv_encoder = lambda_bn(conv_encoder)

            pool_encoder = MaxPooling3D(
                pool_size=(2, 2, 2),
                name='maxpool_{}'.format(i + 1)
            )(conv_encoder)

            if self.verbose:
                print(conv_encoder, pool_encoder)

            pools.append(pool_encoder)
            convs.append(conv_encoder)
            list_num_features.append(num_channel)

        # center connection
        conv_center = self._conv(
            pools[-1],
            filters=list_num_features[-1],
            kernel_size=3,
            padding='same',
            name='conv_center'
        )

        # residual connection
        conv_center = keras_add([pools[-1], conv_center], name='add_center')

        if self.verbose:
            print(conv_center)

        # decoder steps
        conv_decoders = [conv_center]

        for i in range(1, self.num_poolings + 1):
            decoder_upsample = UpSampling3D(
                size=(2, 2, 2),
                name='upsample_{}'.format(i + 1)
            )(conv_decoders[-1])
            up_decoder = concatenate(
                [decoder_upsample, convs[-i]],
                name='cat_{}'.format(i)
            )
            conv_decoder = up_decoder

            for j in range(self.num_conv_per_pooling):
                conv_decoder = self._conv(
                    conv_decoder,
                    filters=list_num_features[-i],
                    kernel_size=3,
                    padding='same',
                    activation='relu',
                    name='conv_dec_{}_{}'.format(i + 1, j)
                )
                conv_decoder = lambda_bn(conv_decoder)

            conv_decoders.append(conv_decoder)

            if self.verbose:
                print(conv_decoder, up_decoder)

        # output layer

        conv_decoder = conv_decoders[-1]

        conv_output = self._conv(
            conv_decoder,
            self.num_channel_output,
            kernel_size=1,
            padding='same',
            activation=self.final_activation,
            name='model_output'
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
