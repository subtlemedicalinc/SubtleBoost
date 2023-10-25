import torch
import torch.nn as nn
import numpy as np

from subtle.dnn.generators.base import GeneratorBase

class GeneratorUNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'unet2d'
        super().__init__(**kwargs)

        ip_channels = self.num_channel_input

        for i in np.arange(self.num_poolings):
            num_filters = self.num_filters_first_conv * (2 ** i)

            for j in np.arange(self.num_conv_per_pooling):
                layer_name = f'conv_enc_{i}_{j}'
                kernel_size = self.get_config('kernel_size', layer_name)
                padding = self.get_config('padding', layer_name)

                conv_enc = nn.Conv2d(ip_channels, num_filters, tuple(kernel_size), padding=padding)
                ip_channels = num_filters
                setattr(self, layer_name, conv_enc)

                relu_enc= nn.ReLU(inplace=False)
                setattr(self, 'act_{}'.format(layer_name), relu_enc)

            layer_name = f'maxpool_{i}'
            maxpool = nn.MaxPool2d(self.get_config('pool_size', layer_name))
            setattr(self, layer_name, maxpool)

        layer_name = 'conv_center'
        kernel_size = self.get_config('kernel_size', layer_name)
        padding = self.get_config('padding', layer_name)
        conv_center = nn.Conv2d(num_filters, num_filters, tuple(kernel_size), padding=padding)
        setattr(self, layer_name, conv_center)

        layer_name = 'act_conv_center'
        relu_conv_center = nn.ReLU(inplace=False)
        setattr(self, layer_name, relu_conv_center)

        ip_channels = num_filters * 2
        for i in np.arange(self.num_poolings):
            layer_name = f'upsample_{i}'
            upsample = nn.Upsample(
                scale_factor=tuple(self.get_config('upsample_size', layer_name))
            )
            setattr(self, layer_name, upsample)

            for j in np.arange(self.num_conv_per_pooling):
                layer_name = f'conv_dec_{i}_{j}'
                kernel_size = self.get_config('kernel_size', layer_name)
                padding = self.get_config('padding', layer_name)

                conv_dec = nn.Conv2d(ip_channels, num_filters, tuple(kernel_size), padding=padding)
                ip_channels = num_filters
                setattr(self, layer_name, conv_dec)

                relu_dec = nn.ReLU(inplace=False)
                setattr(self, 'act_{}'.format(layer_name), relu_dec)

            num_filters = num_filters // 2
            ip_channels = ip_channels + num_filters

        layer_name = 'model_output'
        kernel_size = self.get_config('kernel_size', layer_name)
        padding = self.get_config('padding', layer_name)

        conv_output = nn.Conv2d(
            num_filters * 2, self.num_channel_output, tuple(kernel_size), padding=padding
        )
        setattr(self, layer_name, conv_output)

        if self.get_config('activation', layer_name) == 'relu':
            relu_output = nn.ReLU(inplace=False)
            setattr(self, 'act_{}'.format(layer_name), relu_output)

    def _conv(self, x, filters, kernel_size=None, padding=None, activation=None, name=None):
        conv_layer = getattr(self, name)
        out = conv_layer(x)

        if hasattr(self, 'act_{}'.format(name)):
            if 'output' in name:
                print('relu-ing output')
            act_fn = getattr(self, 'act_{}'.format(name))
            return act_fn(out)
        return out

    def _upsample(self, dec_inp, enc_inp, idx, name_prefix=''):
        ups_lname = '{}upsample_{}'.format(name_prefix, idx)

        us_layer = getattr(self, ups_lname)
        decoder_upsample = us_layer(dec_inp)

        up_decoder = torch.cat([decoder_upsample, enc_inp], dim=1)
        return up_decoder

    def _encoder_decoder(self, inputs, name_prefix=''):
        # step 1
        conv1 = inputs

        # encoder pools
        convs = [inputs]
        pools = [inputs]
        list_num_features = [self.num_channel_input]

        for i in range(self.num_poolings):
            conv_encoder = pools[-1]
            num_channel = self.num_filters_first_conv * (2**i) # double channels

            for j in range(self.num_conv_per_pooling):
                conv_encoder = self._conv(
                    conv_encoder,
                    filters=num_channel,
                    name='{}conv_enc_{}_{}'.format(name_prefix, i, j)
                )

            maxpool_name = '{}maxpool_{}'.format(name_prefix, i)
            pool_enc_layer = getattr(self, maxpool_name)
            pool_encoder = pool_enc_layer(conv_encoder)

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

        for i in range(self.num_poolings):
            up_decoder = self._upsample(
                conv_decoders[-1], convs[-(i+1)], i, name_prefix=name_prefix
            )
            conv_decoder = up_decoder

            for j in range(self.num_conv_per_pooling):
                conv_decoder = self._conv(
                    conv_decoder,
                    filters=list_num_features[-(i+1)],
                    name='{}conv_dec_{}_{}'.format(name_prefix, i, j)
                )

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
