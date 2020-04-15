import tensorflow as tf
import numpy as np
import cv2
import math
from bvs.utils.create_preds_seq import create_multi_frame
from bvs.utils.create_preds_seq import create_multi_frame_from_multi_channel


class BotUpSaliency(tf.keras.layers.Layer):

    def __init__(self, ksize,
                 K,
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

    def build(self, input_shape):
        # build interconnections kernels, shape: (ksize[0], ksize[1], K, K)  # todo change to (K, ksize[0], ksize[1], K, num_outputs+1)? -> figure out conv4D
        self.W, self.J = self._build_interconnection()
        self.i_norm_k = self._build_i_norm_k()

        self.W = tf.convert_to_tensor(self.W, dtype=tf.float32, name='inhibition_kernel')
        self.J = tf.convert_to_tensor(self.J, dtype=tf.float32, name='excitatory_kernel')
        self.i_norm_k = tf.convert_to_tensor(self.i_norm_k, dtype=tf.float32, name='i_norm_k')

    def call(self, input):
        x = tf.zeros_like(input)
        y = tf.zeros_like(input)

        for t in range(1):

            print()
            print("----------------------------------------------------------")
            print("t", t)
            print("shape W", tf.shape(self.W))
            print("shape J", tf.shape(self.J))
            # i_noise_y = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) / 10 + 0.1
            # i_noise_x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) / 10 + 0.1
            i_noise_y = 0
            i_noise_x = 0

            # if self.verbose >= 4:
            #     self._save_multi_frame(x, "bvs/video/"+str(t)+"_01_x.jpeg")
            #     self._save_multi_frame(self._gx(x)[0], "bvs/video/"+str(t)+"_02_gx(x)_response.jpeg")
            i_norm = 0.85 - 2 * tf.pow(tf.divide(tf.nn.conv2d(self._gx(x), self.i_norm_k, strides=1, padding='SAME'),
                                                 (tf.pow(tf.shape(self.i_norm_k, out_type=self.i_norm_k.dtype)[0], 2))),
                                       2)

            inhibs = []
            excits = []
            for i in range(self.K):
                inhib = tf.nn.conv2d(self._gx(x), self.W[i], strides=1, padding='SAME')
                excit = tf.nn.conv2d(self._gx(x), self.J[i], strides=1, padding='SAME')
                inhibs.append(inhib)
                excits.append(excit)
            #
            # print("shape inhibs", np.shape(inhibs))
            # inhibs = np.swapaxes(np.expand_dims(np.squeeze(inhibs), axis=3), 3, 0)
            # excits = np.swapaxes(np.expand_dims(np.squeeze(excits), axis=3), 3, 0)
            #
            # if save_intermediate_img:
            #     # save i_norm
            #     i_norm_print = np.expand_dims(i_norm[0], axis=2)
            #     i_norm_print = np.array(i_norm_print).astype(np.uint8)
            #     multi_frame = create_multi_frame(i_norm_print, num_row, num_column, (256, 256))
            #     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            #     cv2.imwrite("bvs/video/"+str(t)+"_03_i_norm.jpeg", heatmap.astype(np.uint8))
            #     # save inhibition
            #     inhibs_print = np.expand_dims(inhibs[0], axis=2)
            #     inhibs_print = np.array(inhibs_print).astype(np.uint8)
            #     num_filters = np.shape(inhibs)[-1]
            #     num_column = min(num_filters, max_column)
            #     num_row = math.ceil(num_filters / num_column)
            #     multi_frame = create_multi_frame(inhibs_print, num_row, num_column, (256, 256))
            #     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            #     cv2.imwrite("bvs/video/"+str(t)+"_04_inibition_response.jpeg", heatmap.astype(np.uint8))
            #     # save excitation
            #     excits_print = np.expand_dims(excits[0], axis=2)
            #     excits_print = np.array(excits_print).astype(np.uint8)
            #     multi_frame = create_multi_frame(excits_print, num_row, num_column, (256, 256))
            #     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            #     cv2.imwrite("bvs/video/"+str(t)+"_05_exitatory_response.jpeg", heatmap.astype(np.uint8))
            #
            # print("[convolution] 04")
            # print("[convolution] shape inhibs", np.shape(inhibs))
            # print("[convolution] min max inhibs", np.min(inhibs), np.max(inhibs))
            # print("[convolution] 05")
            # print("[convolution] shape excit", np.shape(excits))
            # print("[convolution] min max excit", np.min(excits), np.max(excits))
            # print()
            #
            # # neural response
            # ####################################################################################################################
            # # y = -alphaY * y + gx(x) + inhibs + Ic + i_noise_y
            # y += epsilon * (-alphaY * y + gx(x) + inhibs + Ic + i_noise_y)
            # ####################################################################################################################
            # print("[Y response] 06")
            # print("[Y response] shape y", np.shape(y))
            # print("[Y response] min max y", np.min(y), np.max(y))
            # print("[Y response] 07")
            # print("[Y response] shape gy(y)", np.shape(gy(y)))
            # print("[Y response] min max gy(y)", np.min(gy(y)), np.max(gy(y)))
            # print()
            #
            # if save_intermediate_img:
            #     # print y neuronal response (inhibitory)
            #     y_print = np.expand_dims(y[0], axis=2)
            #     multi_frame = create_multi_frame(y_print, num_row, num_column, (256, 256))
            #     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            #     cv2.imwrite("bvs/video/"+str(t)+"_06_y_responses.jpeg", heatmap.astype(np.uint8))
            #     # print gy(y)
            #     gy_print = np.expand_dims(gy(y)[0], axis=2)
            #     multi_frame = create_multi_frame(gy_print, num_row, num_column, (256, 256))
            #     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            #     cv2.imwrite("bvs/video/"+str(t)+"_07_gy(y)_response.jpeg", heatmap.astype(np.uint8))
            #
            # # build inhibitory psi matrix
            # inhibs_psi = np.zeros(np.shape(y))
            # print("[psi matrix] 08")
            # print("[psi matrix] shape inhib_psi", np.shape(inhibs_psi))
            # for th in range(n_rot):
            #     psi_tmp = []
            #     for t_p in range(n_rot):
            #         if th != t_p:
            #             theta = th * np.pi / n_rot
            #             theta_p = t_p * np.pi / n_rot
            #             a = np.abs(theta - theta_p)
            #             dth = min(a, np.pi - a)
            #
            #             psi_tmp.append(psi(dth, n_rot) * gy(y[:, :, :, t_p]))
            #     inhibs_psi[:, :, :, th] = np.sum(psi_tmp, axis=0)
            # print("[psi matrix] min man inhibs_psi", np.min(inhibs_psi), np.max(inhibs_psi))
            #
            # # save inhib psi
            # inhibs_psi_print = np.expand_dims(inhibs_psi[0], axis=2)
            # multi_frame = create_multi_frame(inhibs_psi_print, num_row, num_column, (256, 256))
            # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            # cv2.imwrite("bvs/video/"+str(t)+"_08_inibition_psi.jpeg", heatmap.astype(np.uint8))
            #
            # ####################################################################################################################
            # x_inhib = -alphaX * x - gy(y) - inhibs_psi
            # x_excit = J0 * gx(x) + excits + I_i_theta + i_norm + i_noise_x
            # # x = -alphaX * x - gy(y) - inhib_psi + J0 * gx(x) + excit  # term, I_{I, theta} and I0
            # # x = x_inhib + x_excit
            # x += epsilon * (x_inhib + x_excit)  # that what I understood from Zhaoping's li code
            # ####################################################################################################################
            # print("[X response] 09 min max x_inhib", np.min(x_inhib), np.max(x_inhib))
            # print("[X response] 10 min max x_excit", np.min(x_excit), np.max(x_excit))
            # print("[X response] 11 min max x", np.min(x), np.max(x))
            #
            #
            # if save_intermediate_img:
            #     # plot V1 response
            #     x_inhib_print = np.expand_dims(x_inhib[0], axis=2)
            #     multi_frame = create_multi_frame(x_inhib_print, num_row, num_column, (256, 256))
            #     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            #     cv2.imwrite("bvs/video/"+str(t)+"_09_x_inhib_response.jpeg", heatmap.astype(np.uint8))
            #
            #     x_excit_print = np.expand_dims(x_excit[0], axis=2)
            #     multi_frame = create_multi_frame(x_excit_print, num_row, num_column, (256, 256))
            #     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            #     cv2.imwrite("bvs/video/"+str(t)+"_10_x_excit_response.jpeg", heatmap.astype(np.uint8))
            #
            # x_print = np.expand_dims(x[0], axis=2)
            # # x_print = x_print - np.min(x_print)
            # # x_print = x_print / np.max(x_print)
            # # x_print[x_print < 0] = 0
            # print("min max x_print", np.min(x_print), np.max(x_print))
            # multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
            # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            # cv2.imwrite("bvs/video/"+str(t)+"_11_V1_x_response.jpeg", heatmap.astype(np.uint8))
            #
            # x_print = np.expand_dims(gx(x)[0], axis=2)
            # multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
            # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
            # cv2.imwrite("bvs/video/"+str(t)+"_12_V1_gx(x)_response.jpeg", heatmap.astype(np.uint8))

        return input, x

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

                        # compute W connections (inhibition)
                        if d != 0 and d < d_max and beta >= self.maxBeta and np.abs(theta1) > self.maxTheta and d_theta < self.maxDTheta:
                            W[i, j, k, dp] = 0.141 * (1 - np.exp(-0.4 * np.power(beta/d, 1.5)))*np.exp(-np.power(d_theta/(np.pi/4), 1.5))

                        if np.abs(theta2) < self.maxTheta2:
                            max_beta_J = self.maxBeta
                        else:
                            max_beta_J = np.pi / 2.69

                        # compute J connections (excitatory)
                        if 0 < d <= 10 and beta < max_beta_J:
                            b_div_d = beta/d
                            J[i, j, k, dp] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d, 2)/90)

        if self.verbose >= 2:
            self._save_multi_frame_from_multi_channel(W, "bvs/video/W_inibition_filter.jpeg")
            self._save_multi_frame_from_multi_channel(J, "bvs/video/J_exitatory_filter.jpeg")

        return W, J

    def _build_i_norm_k(self):
        return np.ones((5, 5, self.K, self.K))

    def _gx(self, x):
        # todo: basically an activation function, move it as a class ?
        tf.where(tf.math.less_equal(x, self.Tx), x, tf.zeros_like(x))
        tx = tf.multiply(tf.ones_like(x), tf.constant(self.Tx, dtype=x.dtype))
        tf.where(tf.math.greater(x, self.Tx), x, tf.subtract(x, tx))
        tf.where(tf.math.greater(x, 1), x, tf.ones_like(x))
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

    def _save_multi_frame(self, x, name):
        num_filters = np.shape(x)[-1]
        num_column = min(num_filters, self.max_column)
        num_row = math.ceil(num_filters / num_column)
        x_print = np.expand_dims(x[0], axis=2)
        multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(name, heatmap.astype(np.uint8))

    def _save_multi_frame_from_multi_channel(self, x, name):
        num_input_channel = np.shape(x)[-2]
        num_filters = num_input_channel * np.shape(x)[-1]
        num_column = min(num_filters, self.max_column)
        num_row = math.ceil(num_filters / num_column)
        multi_frame = create_multi_frame_from_multi_channel(x, num_row, num_column, (256, 256), num_input_channel)
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(name, heatmap.astype(np.uint8))










