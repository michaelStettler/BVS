import tensorflow as tf
import numpy as np
import cv2
import math
from bvs.utils.create_preds_seq import create_multi_frame
from bvs.utils.create_preds_seq import create_multi_frame_from_multi_channel


class BotUpSaliency(tf.keras.layers.Layer):

    def __init__(self, ksize,
                 K,
                 steps=16,
                 epsilon=0.01,
                 alphaX=1,
                 alphaY=1,
                 Ic_control=0,
                 J0=0.8,
                 Tx=1,
                 Ly=1.2,
                 g1=0.21,
                 g2=2.5,
                 verbose=0):

        """
        BotUpSaliency (Bottom Up Saliency Map) computes and add a saliency map to the input gabor filter responses.
        The model come from the hypothesis that V1 create a a bottom-up saliency map for preattentive selection and
        segmentation (Understanding Vision, theory, models, and data. By: Zhaoping Li)

        :param ksize:
        :param epsilon:
        :param alphaX:
        :param alphaY:
        :param verbose: 0: no display
                        1: print output
                        2: save output image and kernels
                        3: print intermediate result
                        4: save intermediate image
        """
        super(BotUpSaliency, self).__init__()
        self.ksize = ksize  # kernel size
        self.K = K
        self.steps = steps
        self.epsilon = epsilon
        self.alphaX = alphaX
        self.alphaY = alphaY
        self.Ic_control = Ic_control
        self.Ic = 1 + self.Ic_control
        self.J0 = J0
        self.maxBeta = np.pi / 1.1
        self.maxTheta = np.pi / (K - 0.001)
        self.maxTheta2 = np.pi / (K / 2 - 0.1)
        self.maxDTheta = np.pi / 3 - 0.00001  # -0.0001 it's just to compensate some minor imprecision when using np.pi
        self.Tx = Tx
        self.Ly = Ly
        self.g1 = g1
        self.g2 = g2
        self.verbose = verbose
        self.tol = 0.001  # tolerance for angle computations
        self.max_column = K  # option for saving images

        padX = int(self.ksize[1]/2)
        padY = int(self.ksize[0]/2)
        self.padding = [[0, 0], [padY, padY], [padX, padX], [0, 0]]

    def build(self, input_shape):
        self.W, self.J = self._build_interconnection()
        self.i_norm_kernel = self._build_i_norm_kernel()
        self.psi_kernel = self._build_psi_kernel()

        self.W = tf.convert_to_tensor(self.W, dtype=tf.float32, name='inhibition_kernel')
        self.J = tf.convert_to_tensor(self.J, dtype=tf.float32, name='excitatory_kernel')
        self.i_norm_kernel = tf.convert_to_tensor(self.i_norm_kernel, dtype=tf.float32, name='i_norm_kernel')
        self.psi_kernel = tf.convert_to_tensor(self.psi_kernel, dtype=tf.float32, name='psi_kernel')

    def call(self, inputs, **kwargs):
        # # normalize input # todo should I do it with a BN layer ?
        # inputs = inputs - tf.reduce_min(inputs)
        # inputs = inputs / tf.reduce_max(inputs)

        # init x and y
        x = tf.ones_like(inputs) * 0.01
        gx = self._gx(x)
        y = tf.ones_like(inputs)
        gy = self._gy(y)
        shape_i_norm_k = tf.shape(self.i_norm_kernel, out_type=self.i_norm_kernel.dtype)
        out = tf.zeros_like(inputs)

        # i_noise_y = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) / 10 + 0.1  # todo add paremeter to decide to add noise
        # i_noise_x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) / 10 + 0.1
        i_noise_y = 0
        i_noise_x = 0

        for t in range(self.steps):
            # compute i_norm
            i_norm = 0.85 - 2 * tf.pow(tf.divide(tf.nn.conv2d(gx, self.i_norm_kernel, strides=1, padding='SAME'),
                                                 (tf.pow(shape_i_norm_k[0], 2))), 2)

            # compute excitatory and inhibitory connections
            gx_padded = tf.pad(gx, self.padding, "SYMMETRIC")
            inhib = tf.nn.conv2d(gx_padded, self.W, strides=1, padding='VALID')
            excit = tf.nn.conv2d(gx_padded, self.J, strides=1, padding='VALID')

            # compute psi inhibition
            inhibs_psi = tf.nn.conv2d(gy, self.psi_kernel, strides=1, padding='SAME')

            # compute inhibitory response (interneuron y)
            y += self.epsilon * (-self.alphaY * y + gx + inhib + self.Ic + i_noise_y)

            # compute excitatory repsonse (interneuron x)

            # as ZhaoPing Li's implementation
            # force = i_norm + inputs - self.alphaX * x + self.J0 * gx - gy - inhibs_psi
            # force_excit = force + excit
            # x += self.epsilon * force_excit

            # as me: split between excitatory and inhibitory responses
            x_inhib = self.alphaX * x + gy + inhibs_psi
            x_excit = self.J0 * gx + excit + inputs + i_norm + i_noise_x
            x += self.epsilon * (x_excit - x_inhib)

            # update activations
            gx = self._gx(x)
            gy = self._gy(y)

            out = out + gx

        out /= self.steps

        # for debugg purposes
        # return inputs, x, gx, i_norm, inhib, excit, y, gy, inhibs_psi, force, force_excit, out  # for debug purposes
        # return inputs, x, gx, i_norm, inhib, excit, y, gy, inhibs_psi, x_inhib, x_excit, out  # for debug purposes

        saliency = tf.math.reduce_max(out, axis=3)
        return saliency

    def _build_interconnection(self):
        # declare filters
        W = np.zeros((self.ksize[0], self.ksize[1], self.K, self.K))
        J = np.zeros((self.ksize[0], self.ksize[1], self.K, self.K))

        # compute filters for each orientation (K)
        translate = int(self.ksize[0]/2)
        for k in range(self.K):
            theta = np.pi/2 - k * np.pi / self.K
            for i in range(self.ksize[0]):
                for j in range(self.ksize[1]):
                    # built kernel with center at the middle
                    di = i - translate
                    dj = j - translate
                    if np.abs(di) > 0:
                        alpha = np.arctan(dj/di)
                    elif dj != 0:
                        alpha = np.pi/2
                    else:
                        alpha = 0

                    d = np.sqrt(di**2 + dj**2)

                    for dp in range(self.K):
                        # compute delta theta
                        theta_p = np.pi/2 - dp * np.pi / self.K  # convert dp index to theta_prime in radians
                        a = np.abs(theta - theta_p)
                        d_theta = min(a, np.pi - a)

                        # compute theta1 and theta2 according to the axis from i, j
                        theta1 = theta - alpha
                        theta2 = theta_p - alpha

                        # condition: |theta1| <= |theta2| <= pi/2
                        if np.abs(theta1) > np.pi / 2:  # condition1
                            if theta1 < 0:
                                theta1 += np.pi
                            else:
                                theta1 -= np.pi

                        if np.abs(theta2) > np.pi / 2:  # condition 2
                            if theta2 < 0:
                                theta2 += np.pi
                            else:
                                theta2 -= np.pi

                        if np.abs(theta1) > np.abs(theta2):
                            tmp = theta1
                            theta1 = theta2
                            theta2 = tmp

                        # compute beta
                        beta = 2 * np.abs(theta1) + 2 * np.sin(np.abs(theta1 + theta2))
                        d_max = 10 * np.cos(beta/4)

                        # compute W connections (inhibition)
                        if d != 0 and d < d_max and beta >= self.maxBeta and np.abs(theta1) > self.maxTheta and d_theta < self.maxDTheta:
                            # W[i, j, k, dp] = 0.141 * (1 - np.exp(-0.4 * np.power(beta/d, 1.5)))*np.exp(-np.power(d_theta/(np.pi/4), 1.5))
                            W[i, j, dp, k] = 0.141 * (1 - np.exp(-0.4 * np.power(beta/d, 1.5)))*np.exp(-np.power(d_theta/(np.pi/4), 1.5))  # note the order of k and dp, changed it to fit conv2d

                        if np.abs(theta2) < self.maxTheta2:
                            max_beta_J = self.maxBeta
                        else:
                            max_beta_J = np.pi / 2.69

                        # compute J connections (excitatory)
                        if 0 < d <= 10 and beta < max_beta_J:
                            b_div_d = beta/d
                            # J[i, j, k, dp] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d, 2)/90)
                            J[i, j, dp, k] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d, 2)/90)  # note the order of k and dp, changed it to fit conv2d

        if self.verbose >= 2:
            self._save_multi_frame_from_multi_channel(W, "bvs/video/WBotUp_inibition_filter.jpeg")
            self._save_multi_frame_from_multi_channel(J, "bvs/video/JBotUp_exitatory_filter.jpeg")

        return W, J

    def _build_i_norm_kernel(self):
        return np.ones((5, 5, self.K, self.K))

    def _build_psi_kernel(self):
        psi_kernel = np.zeros((1, 1, self.K, self.K))

        for th in range(self.K):
            for th_p in range(self.K):
                if th != th_p:
                    theta = th * np.pi / self.K
                    theta_p = th_p * np.pi / self.K
                    a = np.abs(theta - theta_p)
                    dth = min(a, np.pi - a)

                    psi_kernel[:, :, th, th_p] = self._psi(dth)
                    # psi_kernel[:, :, th_p, th] = self._psi(dth)

        if self.verbose >= 2:
            self._save_multi_frame_from_multi_channel(psi_kernel, "bvs/video/PsiBotUp_filter.jpeg")

        return psi_kernel

    def _gx(self, x):
        # todo: basically an activation function, move it as a class ?
        x = tf.where(tf.math.less_equal(x, self.Tx), tf.zeros_like(x), x)
        x = tf.where(tf.math.greater(x, self.Tx), x - self.Tx, x)
        x = tf.where(tf.math.greater(x, 1), tf.ones_like(x), x)
        return x

    def _gy(self, y):
        # todo: basically an activation function, move it as a class ?
        y = tf.where(tf.math.less(y, 0), tf.zeros_like(y), y)
        y = tf.where(tf.math.less_equal(y, self.Ly), self.g1 * y, y)
        y = tf.where(tf.math.greater(y, self.Ly), self.g1 * self.Ly + self.g2 * (y - self.Ly), y)
        return y

    def _psi(self, theta):
        theta = np.abs(theta)
        if -self.tol < theta < self.tol:
            return 1
        elif np.pi / self.K - self.epsilon < theta < np.pi / self.K + self.epsilon:
            return 0.8
        elif 2 * np.pi / self.K - self.epsilon < theta < 2 * np.pi / self.K + self.epsilon:
            return 0.7
        else:
            return 0

    def _save_multi_frame_from_multi_channel(self, x, name):
        num_input_channel = np.shape(x)[-2]
        num_filters = num_input_channel * np.shape(x)[-1]
        num_column = min(num_filters, self.max_column)
        num_row = math.ceil(num_filters / num_column)
        multi_frame = create_multi_frame_from_multi_channel(x, num_row, num_column, (256, 256), num_input_channel)
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(name, heatmap.astype(np.uint8))










