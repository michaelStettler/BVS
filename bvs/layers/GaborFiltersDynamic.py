import tensorflow as tf
from bvs.layers import GaborFilters
from tensorflow.keras.layers import Conv2D
import numpy as np
import matplotlib.pyplot as plt


class GaborFiltersDynamic(GaborFilters):
    def __init__(self, ksize,
                 sigma=[1],
                 theta=[np.pi],
                 lamda=[np.pi],
                 gamma=0.5,
                 phi=[0],
                 per_channel=False,
                 per_color_channel=False,
                 use_octave=True,
                 octave=1.6,
                 sigma_t=2,
                 omega=1,
                 psi=0):
        super(GaborFiltersDynamic, self).__init__(ksize,
                                                  sigma,
                                                  theta,
                                                  lamda,
                                                  gamma,
                                                  phi,
                                                  per_channel,
                                                  per_color_channel,
                                                  use_octave,
                                                  octave)

        self.sigma_t = sigma_t
        self.omega = omega
        self.psi = psi
        self.Aplus = 1
        self.Aminus = 0.5

    def build(self, input_shape):
        self.inp_shape = input_shape
        # kernels = np.array([self._build_gabor(self.sigma, theta, self.lambd, self.gamma, self.phi) for theta in self.theta])

        kernels = []
        for phi in self.phi:
            for theta in self.theta:
                for sigma in self.sigmas:
                    for lamda in self.lamdas:
                        gb = self._build_temporal_gabor(sigma,
                                                        theta,
                                                        lamda,
                                                        self.gamma,
                                                        phi,
                                                        self.sigma_t,
                                                        self.omega,
                                                        self.psi)
                        kernels.append(gb)

        kernels = np.moveaxis(kernels, 0, -1)
        kernels = np.expand_dims(kernels, axis=3)

        if self.per_color_channel and self.per_channel:
            print("Parameters mutually exclusive! Please select only one to True")
            print("Set per_color_channel = True and per_channel = False")

        if self.per_color_channel:
            # todo control per_channel for dynamic
            kernels = self._build_color_kernel(input_shape, kernels)
        elif not self.per_channel:  # if per_channel = False
            # todo control color channel for dynamic
            kernels = np.repeat(kernels, input_shape[-1], axis=3)  # repeat on input axis to correlate with output axis
            # of input_shape (create only contrast kernnel)

        self.kernel = tf.convert_to_tensor(kernels, dtype=tf.float32, name='Gabor_kernel')
        print("shape kernel", np.shape(self.kernel))

    def call(self, input):
        if self.per_channel:
            conv = tf.concat([tf.nn.conv3d(tf.expand_dims(input[:, :, :, :, i], axis=4), self.kernel, strides=(1, 1, 1, 1, 1), padding='SAME') for i in range(self.inp_shape[-1])], axis=4)
            return conv
        else:
            return tf.nn.conv3d(input, self.kernel, strides=(1, 1, 1, 1, 1), padding='SAME')

    def _build_temporal_gabor(self, sigma, theta, Lambda, gamma, phi, sigma_t, omega, psi):
        """Construct temporal Gabor Filter (3D) feature extraction."""

        """Gabor feature extraction."""
        sigma_x = sigma
        sigma_y = float(sigma) / gamma
        sigma_t = sigma_t
        k = 2 * np.pi / Lambda
        w = omega
        print("sigma_x", sigma_x, "sigma_t", sigma_t)
        print("k", k, "w", w)

        # Bounding box
        xmax = int(self.ksize[2] / 2)
        ymax = int(self.ksize[1] / 2)
        tmax = int(self.ksize[0] / 2)
        xmin = -xmax
        ymin = -ymax
        tmin = -tmax
        (t, y, x) = np.meshgrid(np.arange(tmin, tmax + 1), np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1), indexing='ij')

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = self.Aplus * np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2 + t ** 2 / sigma_t ** 2)) * \
             np.cos(k * x_theta + w * t + phi + psi) + \
             self.Aminus * np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2 + t ** 2 / sigma_t ** 2)) * \
             np.cos(k * x_theta - w * t + phi - psi)

        # todo: phi(k) and psi(w) ? from formula 2.62 p.46
        # todo: missing weights A+ and A-, but what should they be ?

        return gb









