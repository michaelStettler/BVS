import tensorflow as tf
import math as m
import numpy as np


class GaborFilters(tf.keras.layers.Layer):
    def __init__(self, n_orient, freq, kernel_size=3, sigma=1, beta=2):
        super(GaborFilters, self).__init__()
        self.n_orient = n_orient
        self.freq = freq
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.beta = beta
        self.fi = 1  # Octaves of the half - amplitud bandwidth of the freq response along the optimal orientation

    def build(self, input_shape):
        orientations = np.array(range(self.n_orient)) / self.n_orient * m.pi

        kernels = np.array([self._build_gabor(o, self.freq[0]) for o in orientations])
        self.kernel = tf.convert_to_tensor(kernels, dtype=tf.complex64, name='Gabor_kernel')
    #     TODO make kernel for multiple freq.

    def call(self, input):
        # return tf.matmul(input, self.kernel)
        return input

    def _build_gabor(self, theta, w0):
        # build Gabor kernel
        x = int(round(self.kernel_size * 2 * m.pi / w0))
        x = np.array(range(-x, x + 1))
        X, Y = np.meshgrid(x, x)

        # generate rotated coordinates
        Z1 = X * np.cos(theta) + Y * np.sin(theta)
        Z2 = -X * np.sin(theta) + Y * np.cos(theta)

        # generate frequency - scaled elliptic parameters
        k = m.sqrt(2 * m.log(2)) * (((2 ** self.fi) + 1) / ((2 ** self.fi) - 1))
        sigma = self.sigma * k / w0
        beta = self.beta * sigma

        # compute raw Gabor function
        g_kernel = np.exp(- 0.5 * (np.power(Z1, 2) / sigma ** 2 + np.power(Z2, 2) / beta ** 2))
        g_kernel = g_kernel * np.exp(1j * w0 * Z1)

        # compute DC correction
        DC_kernel = np.exp(- 0.5 * (np.power(Z1, 2) / sigma ** 2 + np.power(Z2, 2) / beta ** 2))
        DC_sum = np.sum(DC_kernel)
        DC_gabor_sum = np.real(np.sum(g_kernel))

        g_kernel = g_kernel - DC_kernel * DC_gabor_sum / DC_sum

        # Normalize to L2 - norm = 1
        L2_g_ker = np.real(np.sum(g_kernel * np.conj(g_kernel)))
        g_kernel = g_kernel / np.sqrt(L2_g_ker)

        # scale for ~k dependency
        g_kernel = g_kernel * w0 / (2 * m.pi)

        return g_kernel











