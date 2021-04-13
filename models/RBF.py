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

    def fit(self, data, verbose=False):
        data = np.transpose(data)

        # compute number of frames -> = number of centers
        # todo here it assumes that each category has the same number of frames
        Nt = data.shape[1] / self.n_category  # =3, Length of TRP got from original code OR len(TRP)

        # initialize centers
        self.centers = data

        # compute rbf function
        size_centers = self.centers.shape
        self.cov_kernel = np.zeros((size_centers[1], size_centers[1]))
        for n in range(0, size_centers[1]):
            for m in range(0, size_centers[1]):
                x_tmp = data[:, n] - self.centers[:, m]
                x_tmp = np.exp(
                    -np.linalg.norm(x_tmp, 2) ** 2 / 2 / self.sigma ** 2)  # norm(var,2)= 2-norm = matlab's norm
                self.cov_kernel[m, n] = x_tmp

        # reorder
        self.kernel = np.zeros((int(Nt), 3, int(Nt), 3))
        for ntp in range(int(self.n_category)):
            for m in range(int(self.n_category)):
                mindex = np.arange(int(Nt)) + m * int(Nt)
                pindex = np.arange(int(Nt)) + ntp * int(Nt)
                mlen = len(mindex)  # not really needed
                plen = len(pindex)  # not really needed
                self.kernel[:, m, :, ntp] = self.cov_kernel[mindex[0]:mindex[mlen - 1] + 1, pindex[0]:pindex[plen - 1] + 1]

        if verbose:
            self.get_response_statistics()

        return self.kernel

    def get_response_statistics(self):
        sig_fire = self.kernel > self.firing_threshold
        sig_fire = np.mean(sig_fire)
        print('[RBF FIT] RBF neurons fire on average for ' + str(sig_fire * 100) + ' % of the training stimuli.')

    def predict(self, data):
        """
        Compute the rbf encoding from trained rbf center to new input

        m := n_frames * n_test_seq
        n := n_neurons * n_train_seq

        :param data: (n_feature, m)
        :return:
        """
        # compute difference between rbf center and input for each frame/neurons
        diff = [[data[:, m] - self.centers[:, n] for m in range(input.shape[1])] for n in range(self.centers.shape[1])]

        # apply gaussian activation to each
        preds = np.exp(-np.linalg.norm(diff, ord=2, axis=2) ** 2 / 2 / self.sigma ** 2)

        return preds

    def plot_rbf_kernel(self, save_folder=None):
        n_neuron = np.shape(self.cov_kernel)[0]
        n_frames = np.shape(self.cov_kernel)[1]

        n_train_seq = np.shape(self.kernel)[1]
        n_test_seq = np.shape(self.kernel)[3]
        seq_length = self.config['batch_size']

        plt.figure()
        im = plt.imshow(self.cov_kernel)
        plt.colorbar(im)
        for m in range(1, n_test_seq):
            # draw vertical line
            plt.plot([m*seq_length, m*seq_length], [0, n_neuron - 1], color='r')
        for n in range(1, n_train_seq):
            # draw hori line
            plt.plot([0, n_frames - 1], [n*seq_length, n*seq_length], color='r')
        plt.title('IT neuron (RBF) responses')
        plt.xlabel('# of stimulus')
        plt.ylabel('Neuron #')
        plt.savefig('RBF_testresponse.png')

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, 'RBF_trainresponses.png'))
        else:
            plt.savefig('RBF_trainresponses.png')
