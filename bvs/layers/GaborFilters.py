import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np

# todo gaborfilter: set the 1.6 octaves wavelength
# todo gaborfilter: colorcoding
# todo gaborfilter: dynamic coding ?


class GaborFilters(tf.keras.layers.Layer):
    def __init__(self, ksize, sigma=[3], theta=[np.pi], lamda=[np.pi], gamma=0.5, phi=[0], per_channel=False):
        super(GaborFilters, self).__init__()
        self.ksize = ksize
        self.sigmas = sigma  # standard deviation of the gaussian envelope
        self.theta = theta  # orientation of the normal to the parallel stripes of a Gabor function
        self.lamdas = lamda  # wavelength of the sinusoidal factor
        self.gamma = gamma  # spatial aspect ratio.
        self.phi = phi  # phase offset
        self.per_channel = per_channel  # apply a Gabor filter per channel

    def build(self, input_shape):
        self.inp_shape = input_shape
        # kernels = np.array([self._build_gabor(self.sigma, theta, self.lambd, self.gamma, self.phi) for theta in self.theta])

        kernels = []
        for phi in self.phi:
            for theta in self.theta:
                for sigma in self.sigmas:
                    for lamda in self.lamdas:
                        gb = self._build_gabor(sigma, theta, lamda, self.gamma, phi)
                        kernels.append(gb)

        kernels = np.swapaxes(kernels, 0, -1)
        kernels = np.expand_dims(kernels, axis=2)
        if not self.per_channel:
            kernels = np.repeat(kernels, input_shape[-1], axis=2)  # repeat on input dim
        print("shape kernels", np.shape(kernels))
        self.kernel = tf.convert_to_tensor(kernels, dtype=tf.float32, name='Gabor_kernel')
    #     TODO make 3D kernel per colors?

    def call(self, input):
        if self.per_channel:
            conv = tf.concat([tf.nn.conv2d(tf.expand_dims(input[:, :, :, i], axis=3), self.kernel, strides=1, padding='SAME') for i in range(self.inp_shape[-1])], axis=3)
            return conv
        else:
            return tf.nn.conv2d(input, self.kernel, strides=1, padding='SAME')

    def _build_gabor(self, sigma, theta, Lambda, gamma, phi):
        """Gabor feature extraction."""
        sigma_x = sigma
        sigma_y = float(sigma) / gamma

        # Bounding box
        xmax = int(self.ksize[0] / 2)
        ymax = int(self.ksize[1] / 2)
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * \
             np.cos(2 * np.pi / Lambda * x_theta + phi)

        return gb










