import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np

# todo gaborfilter: set the 1.6 octaves wavelength
# todo gaborfilter: colorcoding
# todo gaborfilter: dynamic coding ?


class GaborFilters(tf.keras.layers.Layer):
    def __init__(self, ksize,
                 sigma=[1],
                 theta=[np.pi],
                 lamda=[np.pi],
                 gamma=0.5,
                 phi=[0],
                 per_channel=False,
                 per_color_channel=False,
                 use_octave=True,
                 octave=1.6):
        super(GaborFilters, self).__init__()
        self.ksize = ksize
        self.sigmas = sigma  # standard deviation of the gaussian envelope  -> usually computed with the octave
        self.theta = theta  # orientation of the normal to the parallel stripes of a Gabor function
        self.lamdas = lamda  # wavelength of the sinusoidal factor
        self.gamma = gamma  # spatial aspect ratio.
        self.phi = phi  # phase offset
        self.per_channel = per_channel  # apply a Gabor filter per channel (duplicate for RGB channel)
        self.per_color_channel = per_color_channel  # apply a Gabor filter across color channels (duplicate in function
        # of color such that R could be inhibited by G, B and GB

        if use_octave:
            # http: // www.cs.rug.nl / ~imaging / simplecell.html
            self.sigmas = self.lamdas * 1 / np.pi * np.sqrt(np.log(2) / 2) * (np.power(2, octave) + 1) / (np.power(2, octave) - 1)
            print("[computed from Lambda ({})] self.sigmas = {}".format(self.lamdas, self.sigmas))

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

        if self.per_color_channel and self.per_channel:
            print("Parameters mutually exclusive! Please select only one to True")
            print("Set per_color_channel = True and per_channel = False")

        if self.per_color_channel:
            kernels = self._build_color_kernel(input_shape, kernels)
        elif not self.per_channel:  # if per_channel = False
            kernels = np.repeat(kernels, input_shape[-1], axis=2)  # repeat on input axis to correlate with output axis
            # of input_shape (create only contrast kernnel)

        self.kernel = tf.convert_to_tensor(kernels, dtype=tf.float32, name='Gabor_kernel')

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

    def _build_color_kernel(self, input_shape, kernels):
        diff_degree = input_shape[-1] - np.shape(kernels)[2]  # difference between input and output size

        ks = []
        for i in range(input_shape[-1]):  # loop over input size (i.e. RGB)
            k = np.zeros(np.shape(kernels))
            k = np.repeat(k, input_shape[-1], axis=2)  # repeat on input axis to correlate with output axis
            k[:, :, i, :] = kernels[:, :, 0, :]  # init color channel

            if diff_degree == 2:
                k = self._expand_2deg(k, i)
            else:
                # todo sort out how to do for any possibility
                raise NotImplementedError

            ks.append(k)

        # reshape to match tensorflow dimensions
        ks = np.moveaxis(ks, 0, -1)  # first move axis cause reshape does it with channel last first!
        ks = np.reshape(ks, (np.shape(ks)[0], np.shape(ks)[1], np.shape(ks)[2], -1))

        # add contrast kernels (on each channel)
        contrast_kernels = kernels.copy()
        contrast_kernels = np.repeat(contrast_kernels, input_shape[-1], axis=2)
        ks = np.concatenate([ks, contrast_kernels], axis=3)

        return ks

    def _expand_2deg(self, k, i):
        """
        Expand kernel k by two degree -> = 3 new combinations
        :param k:
        :return:
        """
        k0 = k.copy()

        k1 = k.copy()
        k1[:, :, (i+1) % 3, :] = -k[:, :, i, :]  # do only one since then it duplicates

        # k2 = k.copy()
        # k2[:, :, (i+2) % 3, :] = -k[:, :, i, :]

        # k3 = k.copy()
        # k3[:, :, (i+1) % 3, :] = k[:, :, i, :]
        # k3[:, :, (i+2) % 3, :] = k[:, :, i, :]

        # return np.concatenate([k0, k1, k2, k3], axis=3)
        # return np.concatenate([k0, k1, k2], axis=3)
        return np.concatenate([k0, k1], axis=3)









