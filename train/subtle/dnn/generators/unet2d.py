import torch
import torch.nn as nn
import numpy as np

from subtle.dnn.generators.base import GeneratorBase

class GeneratorUNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'unet2d'
        super().__init__(**kwargs)

    def _conv(self, x, filters, kernel_size=None, padding=None, activation=None, name=None):
        activation = activation if activation is not None else self.get_config('activation', name)
        padding = padding if padding is not None else self.get_config('padding', name)
        kernel_size = kernel_size if kernel_size is not None else self.get_config('kernel_size', name)

        conv_layer = nn.Conv2d(
            x.shape[1], filters, tuple(kernel_size), padding=padding
        )
        setattr(self, name, conv_layer)
        out = conv_layer(x)

        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(
                alpha=self.get_config('lrelu_alpha', name),
                inplace=True
            )
        else:
            act_fn = lambda x: x

        setattr(self, '{}_act'.format(name), act_fn)

        return act_fn(out)

    def _upsample(self, dec_inp, enc_inp, idx, name_prefix=''):
        ups_lname = '{}upsample_{}'.format(name_prefix, idx + 1)

        us_layer = nn.Upsample(
            scale_factor=tuple(self.get_config('upsample_size', ups_lname))
        )
        decoder_upsample = us_layer(dec_inp)
        setattr(self, ups_lname, us_layer)

        up_decoder = torch.cat([decoder_upsample, enc_inp], dim=1)
        return up_decoder

    def _encoder_decoder(self, inputs, name_prefix=''):
        # batch norm
        if self.batch_norm:
            lambda_bn = lambda x: nn.BatchNorm2d(x.shape[1])(x)
        else:
            lambda_bn = lambda x: x

        # step 1
        conv1 = inputs

        for i in range(self.num_conv_per_pooling):
            conv1 = self._conv(
                conv1,
                filters=self.num_filters_first_conv,
                name='{}conv_enc_1_{}'.format(name_prefix, i)
            )
            conv1 = lambda_bn(conv1)

        pool1_layer = nn.MaxPool2d(self.get_config('pool_size', 'maxpool_1'))
        pool1 = pool1_layer(conv1)
        setattr(self, 'maxpool_1', pool1_layer)

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
                    name='{}conv_enc_{}_{}'.format(name_prefix, i + 1, j)
                )
                conv_encoder = lambda_bn(conv_encoder)


            maxpool_name = '{}maxpool_{}'.format(name_prefix, i + 1)
            pool_enc_layer = nn.MaxPool2d(
                self.get_config('pool_size', maxpool_name)
            )
            pool_encoder = pool_enc_layer(conv_encoder)
            setattr(self, maxpool_name, pool_enc_layer)


            pools.append(pool_encoder)
            convs.append(conv_encoder)
            list_num_features.append(num_channel)

        # center connection
        conv_center = self._conv(
            pools[-1],
            filters=list_num_features[-1],
            name='{}conv_center'.format(name_prefix)
        )

        # residual connection
        conv_center = torch.add(pools[-1], conv_center)

        # decoder steps
        conv_decoders = [conv_center]

        for i in range(1, self.num_poolings + 1):
            up_decoder = self._upsample(
                conv_decoders[-1], convs[-i], i, name_prefix=name_prefix
            )
            conv_decoder = up_decoder


            for j in range(self.num_conv_per_pooling):
                conv_decoder = self._conv(
                    conv_decoder,
                    filters=list_num_features[-i],
                    name='{}conv_dec_{}_{}'.format(name_prefix, i + 1, j)
                )
                conv_decoder = lambda_bn(conv_decoder)


            conv_decoders.append(conv_decoder)

        conv_decoder = conv_decoders[-1]
        return conv_decoder

    def forward(self, inputs):
        conv_decoder = self._encoder_decoder(inputs)

        conv_output = self._conv(
            conv_decoder,
            filters=self.num_channel_output,
            kernel_size=self.get_config('kernel_size', 'model_output'),
            activation=self.get_config('activation', 'model_output'),
            name='model_output'
        )

        return conv_output
