"""
ConstMultiplierLayer for EDSR

@authors: Long Wang (long@subtlemedical.com)
Copyright (c) Subtle Medical, Inc.
"""

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Constant

class ConstMultiplier(Layer):
    def __init__(self, val=0.1, **kwargs):
        self.val = val
        super(ConstMultiplier, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer=Constant(value=self.val),
            dtype='float32',
            trainable=True,
        )
        super(ConstMultiplier, self).build(input_shape)

    def call(self, x):
        return tf.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape
