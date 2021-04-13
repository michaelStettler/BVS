import tensorflow as tf
import numpy as np


class RBF(tf.keras.layers.Layer):
    """
    own implementation of a Radial Basis Function (RBF)

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

    def build(self, input_shape):
        self.inp_shape = input_shape

    def call(self, input):
        """
        compute RBF with x - x'

        :param input:
        :return:
        """

        # repetition of first dim into new dimension
        x = tf.reshape(tf.tile(input, [tf.shape(input)[0], 1]), [-1, tf.shape(input)[0], tf.shape(input)[1]])
        # repetition of second dim into new dimension
        x_prime = tf.reshape(tf.tile(input, [1, tf.shape(input)[0]]), [-1, tf.shape(input)[0], tf.shape(input)[1]])
        # compute the RBF as gaussian
        rbf = tf.exp(-tf.norm(x - x_prime, axis=2) ** 2 / 2 / self.sigma ** 2)

        return rbf











