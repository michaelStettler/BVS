import tensorflow as tf
import numpy as np
import cv2
import math
from bvs.utils.create_preds_seq import create_multi_frame_from_multi_channel


class BotUpSaliency(tf.keras.layers.Layer):

    def __init__(self, ksize,
                 K,
                 epsilon=0.01,
                 alphaX=1,
                 alphaY=1,
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
        self.epsilon = epsilon
        self.alphaX = alphaX
        self.alphaY = alphaY
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
        self.max_column = K  # printing option for saving images

        self.W, self.J = self._build_interconnection()

    def build(self, input_shape):
        print("build")

    def call(self, input):
        print("call")

    def _build_interconnection(self):
        # declare filters
        W = np.zeros((self.ksize[0], self.ksize[1], self.K, self.K))
        J = np.zeros((self.ksize[0], self.ksize[1], self.K, self.K))

        # compute filters for each orientation (K)
        translate = int(self.ksize[0]/2)
        for k in range(self.K):
            theta = k * np.pi / self.K
            for i in range(self.ksize[0]):
                for j in range(self.ksize[1]):
                    # built kernel with center at the middle
                    di = i - translate
                    dj = j - translate
                    alpha = np.arctan2(-di, dj)  # -di because think of alpha in a normal x,y coordinate and not a matrix
                    # therefore axes i should goes up
                    if np.abs(alpha) > np.pi / 2:
                        if alpha < 0:
                            alpha += np.pi
                        else:
                            alpha -= np.pi
                    d = np.sqrt(di**2 + dj**2)

                    for dp in range(self.K):
                        # compute delta theta
                        theta_p = dp * np.pi / self.K  # convert dp index to theta_prime in radians
                        a = np.abs(theta - theta_p)
                        d_theta = min(a, np.pi - a)
                        # compute theta1 and theta2 according to the axis from i, j
                        theta1 = theta - alpha
                        theta2 = np.pi - (theta_p - alpha)

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
                        # compute beta
                        beta = 2 * np.abs(theta1) + 2 * np.sin(np.abs(theta1 + theta2))
                        d_max = 10 * np.cos(beta/4)
                        if d != 0 and d < d_max and beta >= self.maxBeta and np.abs(theta1) > self.maxTheta and d_theta < self.maxDTheta:
                            W[i, j, k, dp] = 0.141 * (1 - np.exp(-0.4 * np.power(beta/d, 1.5)))*np.exp(-np.power(d_theta/(np.pi/4), 1.5))

                        if np.abs(theta2) < self.maxTheta2:
                            max_beta_J = self.maxBeta
                        else:
                            max_beta_J = np.pi / 2.69

                        if 0 < d <= 10 and beta < max_beta_J:
                            b_div_d = beta/d
                            J[i, j, k, dp] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d, 2)/90)

        if self.verbose >= 2:
            self._save_multi_frame_from_multi_channel(W, "bvs/video/W_inibition_filter.jpeg")
            self._save_multi_frame_from_multi_channel(J, "bvs/video/J_exitatory_filter.jpeg")

        return W, J

    def _gx(self, x):
        # todo: basically an activation function, move it as a class ?
        x = np.array(x)  # todo make it as tensor!
        x[x <= self.Tx] = 0
        x[x > self.Tx] = x[x > self.Tx] - self.Tx
        x[x > 1] = 1
        return x

    def _gy(self, y):
        # todo: basically an activation function, move it as a class ?
        y = np.array(y)  # todo make it as tensor!
        y[y < 0] = 0
        y[y <= self.Ly] = self.g1 * y[y <= self.Ly]
        y[y > self.Ly] = self.g1 * self.Ly + self.g2 * (y[y > self.Ly] - self.Ly)
        return y

    def _psi(self, theta):
        theta = np.abs(theta)   # todo make it as tensor!
        if -self.tol < theta < self.tol:
            return 1
        elif np.pi / self.K - self.epsilon < theta < np.pi / self.K + self.epsilon:
            return 0.8
        elif 2 * np.pi / self.K - self.epsilon < theta < 2 * np.pi / self.K + self.epsilon:
            return 0.7
        else:
            return 0

    def _save_multi_frame_from_multi_channel(self, kernel, name):
        num_input_channel = np.shape(kernel)[-2]
        num_filters = num_input_channel * np.shape(kernel)[-1]
        num_column = min(num_filters, self.max_column)
        num_row = math.ceil(num_filters / num_column)
        multi_frame = create_multi_frame_from_multi_channel(kernel, num_row, num_column, (256, 256), num_input_channel)
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(name, heatmap.astype(np.uint8))










