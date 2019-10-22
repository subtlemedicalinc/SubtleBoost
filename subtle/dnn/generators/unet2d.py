import tensorflow as tf
import keras.models
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Activation, ReLU, LeakyReLU
from keras.layers.merge import add as keras_add

from subtle.dnn.generators.base import GeneratorBase

class GeneratorUNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'unet2d'
        super().__init__(**kwargs)
        self._build_model()

    def _conv(self, x, filters, kernel_size=None, padding=None, activation=None, name=None):
        activation = activation if activation is not None else self.get_config('activation', name)

        padding = padding if padding is not None else self.get_config('padding', name)

        kernel_size = kernel_size if kernel_size is not None else self.get_config('kernel_size', name)

        out = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            name=name
        )(x)

        if activation == 'relu':
            act_name = 'relu_{}'.format(name)
            act_fn = ReLU(name=act_name)
        elif activation == 'leaky_relu':
            act_name = 'lrelu_{}'.format(name)
            act_fn = LeakyReLU(
                alpha=self.get_config('lrelu_alpha', name),
                name=act_name
            )
        else:
            act_name = '{}_{}'.format(activation, name)
            act_fn = Activation(activation, name=act_name)

        return act_fn(out)

    def _build_model(self):
        print('Building {} model...'.format(self.model_name))
        # batch norm
        if self.batch_norm:
            lambda_bn = lambda x: BatchNormalization()(x)
        else:
            lambda_bn = lambda x: x

        # layers
        # 2D input is (rows, cols, channels)

        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input), name='model_input')

        if self.verbose:
            print(inputs)

        # step 1
        conv1 = inputs

        for i in range(self.num_conv_per_pooling):
            conv1 = self._conv(
                conv1,
                filters=self.num_filters_first_conv,
                name='conv_enc_1_{}'.format(i)
            )
            conv1 = lambda_bn(conv1)

        pool1 = MaxPooling2D(
            pool_size=self.get_config('pool_size', 'maxpool_1'),
            name='maxpool_1'
        )(conv1)

        if self.verbose:
            print(conv1, pool1)

        # encoder pools
        convs = [inputs, conv1]
        pools = [inputs, pool1]
        list_num_features = [self.num_channel_input, self.num_filters_first_conv]

        for i in range(1, self.num_poolings):
            conv_encoder = pools[-1]
            num_channel = self.num_filters_first_conv * (2**i) # double channels

            for j in range(self.num_conv_per_pooling):
                conv_encoder = self._conv(
                    conv_encoder,
                    filters=num_channel,
                    name='conv_enc_{}_{}'.format(i + 1, j)
                )
                conv_encoder = lambda_bn(conv_encoder)

            maxpool_name = 'maxpool_{}'.format(i + 1)
            pool_encoder = MaxPooling2D(
                pool_size=self.get_config('pool_size', maxpool_name),
                name=maxpool_name
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
            name='conv_center'
        )

        print('conv center before add', conv_center)
        # residual connection
        conv_center = keras_add([pools[-1], conv_center], name='add_center')

        if self.verbose:
            print('conv center...', conv_center)

        # decoder steps
        conv_decoders = [conv_center]

        for i in range(1, self.num_poolings + 1):
            ups_lname = 'upsample_{}'.format(i + 1)
            decoder_upsample = UpSampling2D(
                size=self.get_config('upsample_size', ups_lname),
                name=ups_lname
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
                    name='conv_dec_{}_{}'.format(i + 1, j)
                )
                conv_decoder = lambda_bn(conv_decoder)

            conv_decoders.append(conv_decoder)

            if self.verbose:
                print(conv_decoder, up_decoder)

        conv_decoder = conv_decoders[-1]

        conv_output = self._conv(
            conv_decoder,
            filters=self.num_channel_output,
            kernel_size=self.get_config('kernel_size', 'model_output'),
            activation=self.get_config('activation', 'model_output'),
            name='model_output'
        )

        if self.verbose:
            print(conv_output)

        # model
        model = keras.models.Model(inputs=inputs, outputs=conv_output)
        # model.summary()

        if self.verbose:
            print(model)

        self.model = model

        if self.compile_model:
            self._compile_model()
