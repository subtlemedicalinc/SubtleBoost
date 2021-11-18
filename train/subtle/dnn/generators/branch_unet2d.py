import numpy as np
import tensorflow as tf
from keras import backend as K
import keras.models
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, MaxPooling2D, UpSampling2D, concatenate, Activation, add, multiply, Average
from keras.layers.merge import add as keras_add

from subtle.dnn.generators.base import GeneratorBase
from subtle.dnn.layers.WeightedAverage import WeightedAverage

class GeneratorBranchUNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'branch_unet2d'
        super().__init__(**kwargs)
        self._build_model()

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

        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input), name='model_input')

        print('inputs', inputs)
        num_mods = self.num_modalities

        if num_mods == 5:
            ip_names = ['t1_pre', 't1_low', 't2', 'fl', 'uad']
            t1_pre, t1_low, t2, flair, uad = [
                Lambda(lambda ip: ip[..., idx::num_mods], name=ip_names[idx])(inputs)
                for idx in np.arange(num_mods)
            ]
        else:
            ip_names = ['t1_pre', 't1_low', 't2', 'fl']
            t1_pre, t1_low, t2, flair = [
                Lambda(lambda ip: ip[..., idx::num_mods], name=ip_names[idx])(inputs)
                for idx in np.arange(num_mods)
            ]


        print('t1 pre', t1_pre)
        print('t1 low', t1_low)
        print('t2', t2)
        print('fl', flair)

        nx, ny = self.img_rows, self.img_cols
        nc = int(t1_pre.shape[-1]) * 2

        def ileaver(X, Y):
            il_list = []
            for idx in np.arange(int(X.shape[-1])):
                il_list.append(X[..., idx][..., None])
                il_list.append(Y[..., idx][..., None])
            return concatenate(il_list)

        # branch_1 = concatenate([t1_pre, t1_low], name='branch_t1')
        branch_1 = Lambda(lambda ip: ileaver(ip[0], ip[1]), name='branch_t1')([t1_pre, t1_low])
        b1_decoder = self._encoder_decoder(branch_1, name_prefix='b1_')
        b1_op = self._conv(b1_decoder,
            filters=self.num_branch_op_channels,
            kernel_size=self.get_config('kernel_size', 'model_output'),
            activation=self.get_config('activation', 'model_output'),
            name='branch1_output'
        )

        # branch_2 = concatenate([t1_pre, t2], name='branch_t2')
        branch_2 = Lambda(lambda ip: ileaver(ip[0], ip[1]), name='branch_t2')([t1_pre, t2])
        b2_decoder = self._encoder_decoder(branch_2, name_prefix='b2_')
        b2_op = self._conv(b2_decoder,
            filters=self.num_branch_op_channels,
            kernel_size=self.get_config('kernel_size', 'model_output'),
            activation=self.get_config('activation', 'model_output'),
            name='branch2_output'
        )

        # branch_3 = concatenate([t1_pre, flair], name='branch_fl')
        branch_3 = Lambda(lambda ip: ileaver(ip[0], ip[1]), name='branch_fl')([t1_pre, flair])
        b3_decoder = self._encoder_decoder(branch_3, name_prefix='b3_')
        b3_op = self._conv(b3_decoder,
            filters=self.num_branch_op_channels,
            kernel_size=self.get_config('kernel_size', 'model_output'),
            activation=self.get_config('activation', 'model_output'),
            name='branch3_output'
        )

        cat_ip = [b1_op, b2_op, b3_op]

        if self.enable_uad_branch:
            # branch_4 = concatenate([t1_pre, uad], name='branch_uad')
            branch_4 = Lambda(lambda ip: ileaver(ip[0], ip[1]), name='branch_uad')([t1_pre, uad])
            b4_decoder = self._encoder_decoder(branch_4, name_prefix='b4_')
            b4_op = self._conv(b4_decoder,
                filters=self.num_branch_op_channels,
                kernel_size=self.get_config('kernel_size', 'model_output'),
                activation=self.get_config('activation', 'model_output'),
                name='branch4_output'
            )
            cat_ip.append(b4_op)

        branch_cat = concatenate(cat_ip, name='branch_cat')

        if self.final_step == 'conv':
            conv_output = self._conv(branch_cat,
                filters=self.num_channel_output,
                kernel_size=self.get_config('kernel_size', 'model_output'),
                activation=self.get_config('activation', 'model_output'),
                name='model_output'
            )
        elif self.final_step == 'enc_dec':
            bcat_decoder = self._encoder_decoder(branch_cat, name_prefix='bcat_')
            conv_output = self._conv(bcat_decoder,
                filters=self.num_channel_output,
                kernel_size=self.get_config('kernel_size', 'model_output'),
                activation=self.get_config('activation', 'model_output'),
                name='model_output'
            )
        elif self.final_step == 'weighted_avg':
            conv_output = WeightedAverage(n_output=len(cat_ip), name='model_output')(cat_ip)
        elif self.final_step == 'simple_weighted':
            ip_array = [t1_pre, t1_low, t2, flair]
            weight_avg = WeightedAverage(n_output=len(ip_array), name='wavg_op')(ip_array)
            weight_dec = self._encoder_decoder(weight_avg, name_prefix='wavg_')
            conv_output = conv_output = self._conv(weight_dec,
                filters=self.num_channel_output,
                kernel_size=self.get_config('kernel_size', 'model_output'),
                activation=self.get_config('activation', 'model_output'),
                name='model_output'
            )
        elif self.final_step == 'multi_channel':
            conv_output = branch_cat
        elif self.final_step.startswith('fusion_single_ch'):
            if self.final_step.split('/')[-1] == 'wavg':
                branch_avg = WeightedAverage(n_output=len(cat_ip), name='wavg_op')(cat_up)
            else:
                branch_avg = Average(name='fusion_boost_avg')(cat_ip)
            fboost = concatenate([t1_pre, branch_avg], name='fboost')
            fboost_dec = self._encoder_decoder(fboost, name_prefix='fbst_')
            conv_output = self._conv(fboost_dec,
                filters=self.num_channel_output,
                kernel_size=self.get_config('kernel_size', 'model_output'),
                activation=self.get_config('activation', 'model_output'),
                name='model_output'
            )
        elif self.final_step.startswith('fusion_t1pre_final_1ch'):
            if self.final_step.split('/')[-1] == 'wavg':
                branch_avg = WeightedAverage(n_output=len(cat_ip), name='wavg_op')(cat_up)
            else:
                branch_avg = Average(name='fusion_boost_avg')(cat_ip)
            sl_idx = int(t1_pre.shape[-1] // 2)
            t1_pre_sl = Lambda(lambda x: x[..., sl_idx][..., None], name='t1_pre_sl')(t1_pre)
            fboost = concatenate([t1_pre_sl, branch_avg], name='fboost')
            fboost_dec = self._encoder_decoder(fboost, name_prefix='fbst_')
            conv_output = self._conv(fboost_dec,
                filters=self.num_channel_output,
                kernel_size=self.get_config('kernel_size', 'model_output'),
                activation=self.get_config('activation', 'model_output'),
                name='model_output'
            )
        elif self.final_step == 'fusion_boost_mc':
            branch_avg = Average(name='fusion_boost_avg')(cat_ip)
            fboost = concatenate([t1_pre, branch_avg], name='fboost')
            fboost_dec = self._encoder_decoder(fboost, name_prefix='fbst_')
            fboost_final_conv = self._conv(fboost_dec,
                filters=1,
                kernel_size=self.get_config('kernel_size', 'model_output'),
                activation=self.get_config('activation', 'model_output'),
                name='fboost_final_conv'
            )

            conv_output = concatenate([b1_op, b2_op, b3_op, fboost_final_conv], name='model_output')
        elif self.final_step == 'fusion_boost_sum':
            branch_sum = keras_add(cat_ip, name='fusion_boost_sum')
            fboost = concatenate([t1_pre, branch_sum], name='fboost')
            fboost_dec = self._encoder_decoder(fboost, name_prefix='fbst_')
            fboost_final_conv = self._conv(fboost_dec,
                filters=1,
                kernel_size=self.get_config('kernel_size', 'model_output'),
                activation=self.get_config('activation', 'model_output'),
                name='fboost_final_conv'
            )

            conv_output = concatenate([b1_op, b2_op, b3_op, fboost_final_conv], name='model_output')

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
