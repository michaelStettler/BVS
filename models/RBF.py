import os
import numpy as np
import matplotlib.pyplot as plt


class RBF:

    def __init__(self, config, sigma=None):
        """
        Radial Basis Function (RFB) implementation


        :param config:
        """

        # declare parameters
        self.config = config
        self.n_category = config['n_category']
        if sigma is None:
            self.sigma = config["rbf_sigma"]
        else:
            self.sigma = sigma
        self.firing_threshold = config["rbf_firing_threshold"]
        self.centers = None  # (n_feature, n)
        self.kernel = None  # (seq_length, n_category, seq_length, n_test_sequence)

    def fit(self, data):
        num_dim = len(np.shape(data))

        # flatten array if not in format (num_data, n_features)
        if num_dim > 2:
            data = np.reshape(data, (len(data), -1))

        # initialize centers (fit)
        self.centers = data
        print("[RBF_pattern FIT] shape centers", np.shape(self.centers))

        # predict RBF_pattern kernel
        return self._compute_rbf_kernel(data)

    def fit2d(self, data):
        """
        Allow to fit 2D templates

        :return:
        """

        # initialize centers (fit)
        self.centers = data

        return self._compute_2d_rbf_kernel(data)

    def reshape_preds(self, preds):
        """
        this function sort the neural field into train/test 4D array

        -> think to modify this in the future?
        :param preds:
        :return:
        """
        seq_length = self.config['seq_length']
        n_sequence = np.shape(preds)[1] // seq_length
        print("[RBF_pattern RESHAPE] shape preds", np.shape(preds))
        print("[RBF_pattern RESHAPE] n_sequence", n_sequence)

        # reorder
        kernel = np.zeros((seq_length, self.n_category, seq_length, n_sequence))
        for n in range(n_sequence):
            for m in range(self.n_category):
                mindex = np.arange(seq_length) + m * seq_length
                pindex = np.arange(seq_length) + n * seq_length
                mlen = len(mindex)  # not really needed
                plen = len(pindex)  # not really needed
                kernel[:, m, :, n] = preds[mindex[0]:mindex[mlen - 1] + 1, pindex[0]:pindex[plen - 1] + 1]

        return kernel

    def get_response_statistics(self, data):
        sig_fire = data > self.firing_threshold
        sig_fire = np.mean(sig_fire)
        print('[RBF_pattern FIT] RBF_pattern neurons fire on average for ' + str(sig_fire * 100) + ' % of the training stimuli.')

    def predict(self, data):
        # compute RBF_pattern kernel according to the centers
        return self._compute_rbf_kernel(data)

    def predict2d(self, data):
        # compute RBF_pattern kernel according to the centers
        return self._compute_2d_rbf_kernel(data)

    def _compute_rbf_kernel(self, data):
        """
        Compute the rbf encoding from trained rbf center to new input

        m := num_data
        n := n_centers

        :param data: (n_feature, m)
        :return:
        """

        # self.cov_kernel = np.zeros((size_centers[1], size_centers[1]))
        # print("[RBF_pattern FIT] shape cov_kernel", np.shape(self.cov_kernel))
        # for n in range(0, size_centers[1]):
        #     for m in range(0, size_centers[1]):
        #         x_tmp = data[:, n] - self.centers[:, m]
        #         x_tmp = np.exp(
        #             -np.linalg.norm(x_tmp, 2) ** 2 / 2 / self.sigma ** 2)  # norm(var,2)= 2-norm = matlab's norm
        #         self.cov_kernel[m, n] = x_tmp

        # compute difference between rbf center and input for each frame/neurons
        diff = [[data[m] - self.centers[n] for m in range(len(data))] for n in range(len(self.centers))]

        # apply gaussian activation to each
        kernel = np.exp(-np.linalg.norm(diff, ord=2, axis=2) ** 2 / 2 / self.sigma ** 2)

        return kernel

    def _compute_2d_rbf_kernel(self, data):
        # get shapes of input data
        n_entry = len(data)
        shape_x = np.shape(data)[1]
        shape_y = np.shape(data)[2]
        n_channels = np.shape(data)[3]
        # print("[RBF_pattern] shape x/y", shape_x, shape_y)

        # get shapes of centers (kernel)
        ker_size = (np.shape(self.centers)[1], np.shape(self.centers)[2])
        # print("[RBF_pattern] ker_size", ker_size)
        padd_x = ker_size[0] // 2
        padd_y = ker_size[1] // 2

        # build padded prediction
        # == 'SAME'
        padd_data = np.zeros(
            (n_entry, shape_x + ker_size[0] - 1, shape_y + ker_size[1] - 1, n_channels))
        padd_data[:, padd_x:padd_x + shape_x, padd_y:padd_y + shape_y, :] = data

        # convolve with the eyebrow template
        diffs = []
        for center in self.centers:
            for x in range(shape_x):
                for y in range(shape_y):
                    patch = padd_data[:, x:x + ker_size[0], y:y + ker_size[1]]

                    diffs.append(patch - np.repeat(np.expand_dims(center, axis=0), len(data), axis=0))

        # compute rbf kernel
        kernels = []
        for diff in diffs:
            diff_ = np.reshape(diff, (len(diff), -1))  # flatten so we could compute the norm on axis 1
            kernels.append(np.exp(-np.linalg.norm(diff_, ord=2, axis=1) ** 2 / 2 / self.sigma ** 2))

        # reshape kernels to fit the dimensions (n_data, ker_x, ker_y, n_centers)
        kernels = np.moveaxis(kernels, -1, 0)
        kernels = np.reshape(kernels, (len(self.centers), n_entry, shape_x, shape_y))
        kernels = np.moveaxis(kernels, 0, -1)

        return kernels

    def plot_rbf_kernel(self, kernel, save_folder=None, title=None):
        print("shape kernel", np.shape(kernel))
        n_neuron = np.shape(kernel)[0]
        n_frames = np.shape(kernel)[1]

        n_cat = self.config['n_category']
        seq_length = self.config['seq_length']
        n_test_seq = n_frames // seq_length

        # set fig size
        fig_height = np.amax([n_neuron/100, 6.4])
        fig_width = np.amax([n_frames/100, 4.8])

        # create figure
        plt.figure(figsize=(fig_height, fig_width))
        im = plt.imshow(kernel)
        plt.colorbar(im)
        for m in range(1, n_test_seq):
            # draw vertical line
            plt.plot([m*seq_length, m*seq_length], [0, n_neuron - 1], color='r')
        for n in range(1, n_cat):
            # draw hori line
            plt.plot([0, n_frames - 1], [n*seq_length, n*seq_length], color='r')
        plt.title('IT neuron (RBF_pattern) responses')
        plt.xlabel('# of stimulus')
        plt.ylabel('Neuron #')

        # set figure title
        fig_title = 'RBF_trainresponses.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, fig_title))
        else:
            plt.savefig(fig_title)
