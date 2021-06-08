from tensorflow import pad
from keras.engine.topology import Layer

'''
  From https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/

  2D Constant Padding
  Attributes:
    - padding: (padding_width, padding_height) tuple
    - constant: int (default = 0)
'''
class ConstantPadding2D(Layer):
    def __init__(self, padding=(1, 1), constant=0, **kwargs):
        self.padding = tuple(padding)
        self.constant = constant
        super(ConstantPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], mode='CONSTANT', constant_values=self.constant)
