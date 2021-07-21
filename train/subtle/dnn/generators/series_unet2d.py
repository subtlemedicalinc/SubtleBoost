import numpy as np
import tensorflow as tf
from keras import backend as K
import keras.models
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, MaxPooling2D, UpSampling2D, concatenate, Activation, add, multiply, Average
from keras.layers.merge import add as keras_add

from subtle.dnn.generators.base import GeneratorBase
from subtle.dnn.layers.WeightedAverage import WeightedAverage
from subtle.dnn.generators.unet2d import GeneratorUNet2D
import pdb

class GeneratorSeriesUNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'series_unet2d'
        super().__init__(**kwargs)
        self._build_model()

    def _transfer_weights(self, dest_model, ckp_file, branch_num=None, kw=None, op_layer=None, freeze=False, num_ip_channels=None):
        ip_ch = num_ip_channels if num_ip_channels is not None else self.submodel_num_channel
        src_model = GeneratorUNet2D(
            num_channel_input=ip_ch, num_channel_output=self.num_channel_output,
            img_rows=self.img_rows, img_cols=self.img_cols,
            verbose=self.verbose,
            compile_model=False,
            model_config='base',
            checkpoint_file=ckp_file
        )
        src_model.load_weights()

        if kw is None and op_layer is None:
            kw = 'g{}_'.format(branch_num)
            op_layer = 'generator{}_output'.format(branch_num)

        print('Transferring weights -> {} from {}'.format(kw, ckp_file))
        # pdb.set_trace()
        src_layers = [l.name for l in src_model.model.layers]
        for idx, layer in enumerate(dest_model.layers):
            if kw not in layer.name:
                continue
            unet_name = layer.name.replace(kw, '')

            if unet_name in src_layers:
                lname = unet_name
                lname_orig = layer.name
                src_weights = src_model.model.layers[src_layers.index(unet_name)].get_weights()
                dest_model.layers[idx].set_weights(src_weights)
                dest_model.layers[idx].trainable = (not freeze)

        dest_model.get_layer(op_layer).set_weights(
            src_model.model.get_layer('model_output').get_weights()
        )
        dest_model.get_layer(op_layer).trainable = (not freeze)
        return dest_model

    def _encoder_decoder(self, inputs, name_prefix=''):
        # batch norm
        if self.batch_norm:
            lambda_bn = lambda x: BatchNormalization()(x)
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

        pool1 = MaxPooling2D(
            pool_size=self.get_config('pool_size', 'maxpool_1'),
            name='{}maxpool_1'.format(name_prefix)
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
                    name='{}conv_enc_{}_{}'.format(name_prefix, i + 1, j)
                )
                conv_encoder = lambda_bn(conv_encoder)

            maxpool_name = '{}maxpool_{}'.format(name_prefix, i + 1)
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
            name='{}conv_center'.format(name_prefix)
        )

        print('conv center before add', conv_center)
        # residual connection
        conv_center = keras_add([pools[-1], conv_center], name='{}add_center'.format(name_prefix))

        if self.verbose:
            print('conv center...', conv_center)

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

            if self.verbose:
                print(conv_decoder, up_decoder)

        conv_decoder = conv_decoders[-1]
        return conv_decoder

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
            act_fn = Activation('relu', name=act_name)
        elif activation == 'leaky_relu':
            act_name = 'lrelu_{}'.format(name)
            act_fn = Activation('leaky_relu',
                alpha=self.get_config('lrelu_alpha', name),
                name=act_name
            )
        else:
            act_name = '{}_{}'.format(activation, name)
            act_fn = Activation(activation, name=act_name)

        return act_fn(out)

    def _upsample(self, dec_inp, enc_inp, idx, name_prefix=''):
        ups_lname = '{}upsample_{}'.format(name_prefix, idx + 1)
        decoder_upsample = UpSampling2D(
            size=self.get_config('upsample_size', ups_lname),
            name=ups_lname
        )(dec_inp)

        if self.upsample_mode == 'attention':
            num_ch = dec_inp.get_shape().as_list()[-1] // 4
            enc_inp = self._attn_block(x=enc_inp, g=decoder_upsample, num_ch=num_ch)

        up_decoder = concatenate(
            [decoder_upsample, enc_inp],
            name='{}cat_{}'.format(name_prefix, idx)
        )

        return up_decoder

    def _build_model(self):
        print('Building {}-{} model...'.format(self.model_name, self.model_config))

        # layers
        # 2D input is (rows, cols, channels)

       
        inputs = Input(shape=(self.img_rows, self.img_cols, 7), name='model_input')

        print('inputs', inputs)

        def ileaver(X, Y):
            il_list = []
            for idx in np.arange(int(X.shape[-1])):
                il_list.append(X[..., idx][..., None])
                il_list.append(Y[..., idx][..., None])
            return concatenate(il_list)

        generator_1 = self._encoder_decoder(inputs, name_prefix='g1_')
        outputs_1 = self._conv(generator_1,
            filters=self.num_branch_op_channels,
            kernel_size=self.get_config('kernel_size', 'model_output'),
            activation=self.get_config('activation', 'model_output'),
            name='generator1_output'
        )
        #pdb.set_trace()
        inputs_2 = Lambda(lambda ip: ileaver(ip[0], ip[1]), name='inputs2')([inputs, outputs_1])
        generator_2 = self._encoder_decoder(inputs_2, name_prefix='g2_')
        outputs_2 = self._conv(generator_2,
                              filters=self.output_channels,
                              kernel_size=self.get_config('kernel_size', 'model_output'),
                              activation=self.get_config('activation', 'model_output'),
                              name='generator2_output'
        )

        if self.verbose:
            print(outputs_2)

        # model
        model = keras.models.Model(inputs=inputs, outputs=outputs_2)
        model.summary()

        if self.verbose:
            print(model)

        if self.transfer_weights:
            print(self.fpaths_pre[0])
            model = self._transfer_weights(model, self.fpaths_pre[0], freeze=True, kw='g1_', op_layer='generator1_output', num_ip_channels=7)

        self.model = model

        if self.compile_model:
            self._compile_model()
