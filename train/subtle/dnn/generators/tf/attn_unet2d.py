import tensorflow as tf
import keras.models
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Activation, Multiply, Lambda
from keras.layers.merge import add as keras_add

from subtle.dnn.generators.base import GeneratorBase

class GeneratorAttnUNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'attn_unet2d'
        super().__init__(**kwargs)
        self._build_model()

    def _attention_block_2d(
        self, input, input_channels=None, output_channels=None, encoder_depth=1, name='at'
    ):
        p = 1
        t = 2
        r = 1

        if input_channels is None:
            input_channels = input.get_shape()[-1].value
        if output_channels is None:
            output_channels = input_channels

        # First Residual Block
        for i in range(p):
            input = self._residual_block_2d(input)

        # Trunc Branch
        output_trunk = input
        for i in range(t):
            output_trunk = self._residual_block_2d(output_trunk)

        # Soft Mask Branch

        ## encoder
        ### first down sampling
        output_soft_mask = MaxPooling2D(padding='same')(input)  # 32x32
        for i in range(r):
            output_soft_mask = self._residual_block_2d(output_soft_mask)

        skip_connections = []
        for i in range(encoder_depth - 1):

            ## skip connections
            output_skip_connection = self._residual_block_2d(output_soft_mask)
            skip_connections.append(output_skip_connection)

            ## down sampling
            output_soft_mask = MaxPooling2D(padding='same')(output_soft_mask)
            for _ in range(r):
                output_soft_mask = self._residual_block_2d(output_soft_mask)

                ## decoder
        skip_connections = list(reversed(skip_connections))
        for i in range(encoder_depth - 1):
            ## upsampling
            for _ in range(r):
                output_soft_mask = self._residual_block_2d(output_soft_mask)
            output_soft_mask = UpSampling2D()(output_soft_mask)
            ## skip connections
            output_soft_mask = keras_add([output_soft_mask, skip_connections[i]])

        ### last upsampling
        for i in range(r):
            output_soft_mask = self._residual_block_2d(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)

        ## Output
        output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
        output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
        output_soft_mask = Activation('sigmoid')(output_soft_mask)

        # Attention: (1 + output_soft_mask) * output_trunk
        output = Lambda(lambda x: x + 1)(output_soft_mask)
        output = Multiply()([output, output_trunk])  #

        # Last Residual Block
        for i in range(p):
            output = self._residual_block_2d(output, name=name)

        return output

    def _residual_block_2d(
        self, input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1,
        name='out'
    ):
        if output_channels is None:
            output_channels = input.get_shape()[-1].value
        if input_channels is None:
            input_channels = output_channels // 4

        strides = (stride, stride)

        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = Conv2D(input_channels, (1, 1))(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(output_channels, (1, 1), padding='same')(x)

        if input_channels != output_channels or stride != 1:
            input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)
        if name == 'out':
            x = keras_add([x, input])
        else:
            x = keras_add([x, input], name=name)
        return x

    def _build_model(self):
        merge_axis = -1
        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input), name='model_input')

        filter_num = self.num_filters_first_conv // 4

        conv1 = Conv2D(filter_num * 4, 3, padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

        pool = MaxPooling2D(pool_size=(2, 2))(conv1)

        res1 = self._residual_block_2d(pool, output_channels=filter_num * 4)
        pool1 = MaxPooling2D(pool_size=(2, 2))(res1)

        res2 = self._residual_block_2d(pool1, output_channels=filter_num * 8)
        pool2 = MaxPooling2D(pool_size=(2, 2))(res2)

        res3 = self._residual_block_2d(pool2, output_channels=filter_num * 16)
        pool3 = MaxPooling2D(pool_size=(2, 2))(res3)

        res4 = self._residual_block_2d(pool3, output_channels=filter_num * 32)
        pool4 = MaxPooling2D(pool_size=(2, 2))(res4)

        res5 = self._residual_block_2d(pool4, output_channels=filter_num * 64)
        res5 = self._residual_block_2d(res5, output_channels=filter_num * 64)

        atb5 = self._attention_block_2d(res4, encoder_depth=1, name='atten1')
        up1 = UpSampling2D(size=(2, 2))(res5)
        merged1 = concatenate([up1, atb5], axis=merge_axis)
        res5 = self._residual_block_2d(merged1, output_channels=filter_num * 32)

        atb6 = self._attention_block_2d(res3, encoder_depth=2, name='atten2')
        up2 = UpSampling2D(size=(2, 2))(res5)
        merged2 = concatenate([up2, atb6], axis=merge_axis)
        res6 = self._residual_block_2d(merged2, output_channels=filter_num * 16)

        atb7 = self._attention_block_2d(res2, encoder_depth=3, name='atten3')
        up3 = UpSampling2D(size=(2, 2))(res6)
        merged3 = concatenate([up3, atb7], axis=merge_axis)
        res7 = self._residual_block_2d(merged3, output_channels=filter_num * 8)

        atb8 = self._attention_block_2d(res1, encoder_depth=4, name='atten4')
        up4 = UpSampling2D(size=(2, 2))(res7)
        merged4 = concatenate([up4, atb8], axis=merge_axis)
        res8 = self._residual_block_2d(merged4, output_channels=filter_num * 4)

        up = UpSampling2D(size=(2, 2))(res8)
        merged = concatenate([up, conv1], axis=merge_axis)

        conv9 = Conv2D(filter_num * 4, 3, padding='same')(merged)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu')(conv9)

        output = Conv2D(self.num_channel_output, 1, padding='same', activation='linear')(conv9)
        model = keras.models.Model(inputs, output)
        # model.summary()

        self.model = model

        if self.compile_model:
            self._compile_model()
