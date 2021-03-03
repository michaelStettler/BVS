import tensorflow as tf
import numpy as np


class MaxPoolDepths(tf.keras.layers.Layer):
    """
    Depth Pooling: Basically 3D max Pooling, but with a prior reshape function of the number of parameters set.
    The layers extend the input axis number by one according to the num_cond parameter such as if:
    input shape (None, None, None, 20) with num_cond = 5 and axis = 3
    Max pool Depths will created a new Tensor of shape (None, None, None, 5, 4) and only then reduce it by max pooling
    applied on the specified axis (=3 in this example), leading to: (None, None, None, 4)
    """
    def __init__(self, ksize, strides, padding, axis, num_cond):
        super(MaxPoolDepths, self).__init__()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.axis = axis
        self.num_cond = num_cond

    def build(self, input_shape):
        self.inp_shape = input_shape

    def call(self, input):
        inp_shape = tf.shape(input)
        num_features = int(np.shape(input)[-1] / self.num_cond)

        input = tf.reshape(input, [inp_shape[0], inp_shape[1], inp_shape[2], self.num_cond, num_features])
        depth_pool = tf.math.reduce_max(
            input, axis=self.axis, keepdims=False, name=None)

        return tf.nn.max_pool2d(
            depth_pool, self.ksize, self.strides, self.padding, data_format='NHWC', name=None)











