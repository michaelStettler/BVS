import tensorflow as tf
import numpy as np


class RBF(tf.keras.layers.Layer):
    """
    own implementation of a RBF network

    use vectorization. Therefore if data is of size (m, n) dataA and dataB is the m times repetitions of the matrix
    into a new dimension to compute each m*m combination (in order to avoid a double for loop over m)

    data (4, 3)
    if data = [[1  5   9]
               [2  6  10]
               [3  7  11]
               [4  8  12]]

    dataA (4, 4, 3)
    dataA = [[[1  5   9]
              [2  6  10]
              [3  7  11]
              [4  8  12]]  # rep 1
              [1  5   9]
              [2  6  10]
              [3  7  11]
              [4  8  12]]  # rep 2
              [1  5   9]
              [2  6  10]
              [3  7  11]
              [4  8  12]]  # rep 3
              [1  5   9]
              [2  6  10]
              [3  7  11]
              [4  8  12]]]  # rep 4

    dataB = [[[1  5   9]
              [1  5  9]
              [1  5  9]
              [1  5  9]]  # rep 1
              [2  6  10]
              [2  6  10]
              [2  6  10]
              [2  6  10]]  # rep 2
              [3  7  11]
              [3  7  11]
              [3  7  11]
              [3  7  12]]  # rep 3
              [4  8  12]
              [4  8  12]
              [4  8  12]
              [4  8  12]]]  # rep 4

    """
    def __init__(self, sigma):
        super(RBF, self).__init__()
        self.sigma = sigma
        print("sigma", self.sigma)

    def build(self, input_shape):
        self.inp_shape = input_shape

    def call(self, input):
        print("shape input", tf.shape(input))
        # repetition of first dim into new dimension
        dataA = tf.matlib.repmat(input, tf.shape(input)[0], 1).reshape(
            (-1, tf.shape(input)[0], tf.shape(input)[1]))
        # repetitions of second dim into new dimensions
        dataB = tf.tile(input, tf.shape(input)[0]).reshape(
            (-1, tf.shape(input)[0], tf.shape(input)[1]))
        rbf2 = tf.exp(-tf.linalg.norm(dataA - dataB, axis=2) ** 2 / 2 / self.sigma ** 2)
        return rbf2











