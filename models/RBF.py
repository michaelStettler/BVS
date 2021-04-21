import os
import numpy as np
import matplotlib.pyplot as plt


class RBF:

    def __init__(self, config):

        # declare parameters
        self.config = config
        self.n_category = config['n_category']
        self.sigma = config["rbf_sigma"]
        self.firing_threshold = config["rbg_firing_threshold"]
        self.centers = None  # (n_feature, n)
        self.cov_kernel = None  # (n_neurons, n_total_frame)
        self.kernel = None  # (seq_length, n_category, seq_length, n_test_sequence)

    def fit(self, data):
        data = np.transpose(data)

        # initialize centers
        self.centers = data

        # test predict
        preds = self._compute_rbf_kernel(data)

        # # compute rbf function
        # size_centers = self.centers.shape
        # self.cov_kernel = np.zeros((size_centers[1], size_centers[1]))
        # print("[RBF FIT] shape cov_kernel", np.shape(self.cov_kernel))
        # for n in range(0, size_centers[1]):
        #     for m in range(0, size_centers[1]):
        #         x_tmp = data[:, n] - self.centers[:, m]
        #         x_tmp = np.exp(
        #             -np.linalg.norm(x_tmp, 2) ** 2 / 2 / self.sigma ** 2)  # norm(var,2)= 2-norm = matlab's norm
        #         self.cov_kernel[m, n] = x_tmp

        return preds

    def reshape_preds(self, preds):
        """
        this function sort the neural field into train/tet 4D array

        -> think to modify this in the future?
        :param preds:
        :return:
        """
        seq_length = self.config['batch_size']
        n_sequence = np.shape(preds)[1] // seq_length
        print("n_sequence", n_sequence)

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
        print('[RBF FIT] RBF neurons fire on average for ' + str(sig_fire * 100) + ' % of the training stimuli.')

    def predict(self, data):
        data = np.transpose(data)

        # compute RBF kernel according to the centers
        preds = self._compute_rbf_kernel(data)

        return preds

    def _compute_rbf_kernel(self, data):
        """
        Compute the rbf encoding from trained rbf center to new input

        m := n_frames * n_test_seq
        n := n_neurons * n_train_seq

        :param data: (n_feature, m)
        :return:
        """
        # compute difference between rbf center and input for each frame/neurons
        diff = [[data[:, m] - self.centers[:, n] for m in range(data.shape[1])] for n in range(self.centers.shape[1])]

        # apply gaussian activation to each
        kernel = np.exp(-np.linalg.norm(diff, ord=2, axis=2) ** 2 / 2 / self.sigma ** 2)

        return kernel

    def plot_rbf_kernel(self, kernel, save_folder=None, title=None):
        print("shape kernel", np.shape(kernel))
        n_neuron = np.shape(kernel)[0]
        n_frames = np.shape(kernel)[1]

        n_cat = self.config['n_category']
        seq_length = self.config['batch_size']
        n_test_seq = n_frames // seq_length

        plt.figure(figsize=(n_frames/100, n_neuron/100))
        im = plt.imshow(kernel)
        plt.colorbar(im)
        for m in range(1, n_test_seq):
            # draw vertical line
            plt.plot([m*seq_length, m*seq_length], [0, n_neuron - 1], color='r')
        for n in range(1, n_cat):
            # draw hori line
            plt.plot([0, n_frames - 1], [n*seq_length, n*seq_length], color='r')
        plt.title('IT neuron (RBF) responses')
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
