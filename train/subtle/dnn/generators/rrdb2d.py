"""
Based on RRDB implementation in https://github.com/rajatkb/RDNSR-Residual-Dense-Network-for-Super-Resolution-Keras
"""

import keras.models
from keras.layers import Input, Conv2D, Concatenate
from keras.layers.merge import add as keras_add

from subtle.dnn.generators.base import GeneratorBase
from subtle.dnn.layers.Subpixel import Subpixel
from keras.optimizers import Adam


class GeneratorRRDB2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'rrdb2d'
        super().__init__(**kwargs)

        self._build_model()

    def _conv(
        self, x, features=None, activation='relu', padding='same', kernel_size=(3, 3),
        strides=(1, 1), name=None
    ):
        return Conv2D(
            filters=features, kernel_size=kernel_size, strides=strides, padding=padding,
            activation=activation, name=name
        )(x)

    def _rdblock(self, x, rd_idx):
        li = [x]
        pas = self._conv(x, self.num_rdb_filters, name='conv_rdb{}_0'.format(rd_idx))

        for idx in range(1, self.num_rd_iters):
            li.append(pas)
            out = Concatenate(axis=3, name='cat_rdb{}_{}'.format(rd_idx, idx))(li)
            pas = self._conv(out, self.num_rdb_filters, name='conv_rdb{}_{}'.format(rd_idx, idx))

        li.append(pas)
        out = Concatenate(axis=3, name='cat_rdb{}_{}'.format(rd_idx, idx+1))(li)
        feat = self._conv(
            out, self.num_rdb_filters * 2, kernel_size=(1, 1),
            name = 'conv_rdb{}_local'.format(rd_idx)
        )

        return keras_add([feat, x], name='add_rdb{}'.format(rd_idx))

    def _build_model(self):
        print('Building RRDB 2D model...')
        print('config vals', self.num_rd_iters, self.num_rd_blocks, self.num_rdb_filters)
        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input), name='model_input')

        if self.verbose:
            print(inputs)

        pass1 = self._conv(inputs, self.num_rdb_filters * 2, name='conv_pass1')
        pass2 = self._conv(pass1, self.num_rdb_filters * 2, name='conv_pass2')

        rdb = self._rdblock(pass2, 0)
        rdb_list = [rdb]

        for idx in range(1, self.num_rd_blocks):
            rdb = self._rdblock(rdb, idx)
            rdb_list.append(rdb)

        out = Concatenate(axis=3, name='cat_rdb_list')(rdb_list)
        out = self._conv(
            out, self.num_rdb_filters * 2, kernel_size=(1, 1), name='conv_post_rdb_1',
            activation=None
        )
        out = self._conv(out, self.num_rdb_filters * 2, name='conv_post_rdb_2', activation=None)

        conv_output = keras_add([out, pass1], name='final_add')
        # conv_output = Subpixel(
        #     self.num_rdb_filters * 2, (3, 3), r=1, padding='same', activation='relu',
        #     name='final_subpixel'
        # )(conv_output)
        conv_output = self._conv(conv_output, self.num_channel_output, activation=None)

        # model
        model = keras.models.Model(inputs=inputs, outputs=conv_output)
        model.summary()

        if self.verbose:
            print(model)

        self.model = model

        if self.compile_model:
            optim = Adam(lr=1e-4, decay=5e-5, amsgrad=False)
            self._compile_model(custom_optim=optim)
