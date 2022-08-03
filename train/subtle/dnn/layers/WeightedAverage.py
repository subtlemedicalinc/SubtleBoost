"""
Weighted average layer for GeneratorBranchUNet2D - multi contrast model
From https://stackoverflow.com/questions/62595660/weighted-average-custom-layer-weights-dont-change-in-tensorflow-2-2-0/62595957#62595957
"""
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Concatenate

class WeightedAverage(Layer):

    def __init__(self, n_output, name=None):
        super(WeightedAverage, self).__init__()
        self.name = name
        self.W = tf.Variable(
            initial_value=tf.random.uniform(shape=[1,1,n_output], minval=0, maxval=1), trainable=True
        ) # (1,1,n_inputs)

    def call(self, inputs):
        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [tf.expand_dims(i, -1) for i in inputs]
        inputs = Concatenate(axis=-1)(inputs) # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1) # (1,1,n_inputs)
        # weights sum up to one on last dim

        return tf.reduce_sum(weights*inputs, axis=-1) # (n_batch, n_feat)
