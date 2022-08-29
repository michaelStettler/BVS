import os
import numpy as np
import matplotlib.pyplot as plt


class Amari:
    """
    Class that implement the Amari neural field model

    Parameters to modify to fine tune the neural field (in order of importance)
        - Aker: Amplitude of the kernel, you should try to make your field activated on the diagonal, the more amplitude
        you will add, the more the diagonal will start to lurch
        - Bker: offset of the amplitude kernel, it allows to make the kernel inhibitory outside its peak
        - dker: tune the speed of the traveling pulse, it will correct the lurching, to test sequence selectivity,
        set the parameter to negative and control that your field is well sequence selective

    Once you have nice travelling wave, you can tweak the following parameter:
        - winh: this is how strong the pattern will inhibit the others

    Kernel normalization:
    Samp/s_alpha/s_beta: amplitude and normalization of the fourier kernel, in our case an amplitude of the kernel
    around 75 was good, the goal is to try to flatten the kernel but keeping it's normalization amplitude somehow
    similar across the fields
    """
    def __init__(self, config):
        self.config = config
        self.seq_length = config['seq_length']
        self.n_category = config['n_category']

        # declare neural field tuning parameters
        self.Aker = config["Amari_Aker"]                # kernel amplitude
        self.Bker = config["Amari_Bker"]                # kernel offset
        self.Dker = config["Amari_Dker"]                # asymmetric shift
        self.w_inh = config["Amari_w_inh"]              # cross-pattern inhibition
        self.sigs = config["Amari_sigs"]                # gaussian luring of input distribution
        self.h = config["Amari_h"]                      # resting level
        self.tau = config["Amari_tau"]                  # time constant
        self.tau_v = 4                                  # time constant (read out neurons)
        self.s_amp = config["Amari_nker_s_amp"]         # amplitude factor of normalization
        self.s_alpha = config["Amari_nker_s_alpha"]     # minimum normalization factor
        self.s_beta = config["Amari_nker_s_beta"]       # blending parameters
        self.sig_ker = config["Amari_ker_sig"]          # interaction kernel

        # declare neural field parameters
        self.xs = np.arange(0, self.seq_length)
        self.xsft = np.floor(self.seq_length / 2).astype(int)
        self.hsft = None                                # todo ask martin what these variablessss name could be
        self.wx = None                                  # interaction kernel
        self.fftwx = None                               # interaction kernel (Fourier domain)
        self.sf_integral = None                         # ...
        self.sf_integral_norm = None                    # ...
        self.sx = None                                  # smoothing kernel
        self.fftsx = None                               # smoothing kernel (Fourier domain)

        # initialize kernels
        self._build_inter_kernel()
        self._build_smooth_input_kernel()

    def _build_inter_kernel(self):
        """
        create the interaction kernel

        :return:
        """
        # create kernel
        self.hsft = 0 * self.xs
        self.hsft[self.xsft] = 1
        self.hsft = self.hsft.reshape(1, self.hsft.shape[0])

        # build interaction kernel
        self.wx = self.Aker * np.exp(-(self.xs - self.xsft - self.Dker) ** 2 / 2 / self.sig_ker ** 2) + self.Bker
        self.wx = self.wx.reshape(1, self.wx.shape[0])
        self.wx = np.round(self.wx, 4)
        self.wx = np.fft.ifft(np.multiply(np.fft.fft(self.wx), np.conj(np.fft.fft(self.hsft))))  # trick to center the kernel
        self.fftwx = np.fft.fft(np.transpose(self.wx), axis=0)

    def _build_smooth_input_kernel(self):
        """
        create the smoothing kernel. This is used to convolved with the input as to smooth it.

        :return:
        """
        # create input smoothing kernel
        self.sx = np.exp(-(self.xs - self.xsft) ** 2 / 2 / self.sigs ** 2)
        self.sx = np.fft.ifft(np.multiply(np.fft.fft(self.sx), np.conj(np.fft.fft(self.hsft))))  # trick to center the kernel

        # fourrier transform of the smoothing input kernel
        self.fftsx = np.fft.fft(np.transpose(self.sx), axis=0)

    def normalize_fft_kernel(self, s, epsilon=.1):
        """
        normalize the fourrier kernel as

        norm := (1 - b)/a + b / (e + sum(s))
        s_norm := A * s * norm

        The goal is to flatten the integral over the field since the intergration may add to the amplitude due to outliers
        from the peak.
        This is mainly visible when a snapchot neurons fires to multiple frame since they appear several times in the
        sequence (think of neutral frames)

        beta allows to blend between a fully 1/alpha norm to a fully 1/sum(s)
        epsilon is here to avoid a division by zero but could be also fine tune
        the amplitude A allows to boost the kernel after normalization
        :return:
        """
        # OLD -> made no mathematical sense to use the squared root
        # norm_Sf_tmp = np.squeeze(Samp * Sff_tmp / (2 + (np.sum(Sff_tmp)) ** 0.5))

        norm = ((1 - self.s_beta) / self.s_alpha) + (self.s_beta / (epsilon + np.sum(s)))
        s_norm = np.squeeze(self.s_amp * s * norm)

        return s_norm

    def predict_neural_field(self, data):
        """
        initialize the neural field
        :return:
        """
        print("[PRED NN] shape data", np.shape(data))
        n_test_seq = np.shape(data)[-1]

        #    initialize neural field and define UF, UFA, Sf_tmp
        Uf0 = - np.ones((self.seq_length, self.n_category)) * self.h

        UF = np.zeros((self.seq_length, self.n_category, self.seq_length))
        UF[:, :, 0] = Uf0  # initialization for time 1
        UFA = np.zeros((self.seq_length, self.n_category, self.seq_length, n_test_seq))
        Sf_tmp = np.zeros((self.seq_length, self.n_category, self.seq_length))

        #    iterate neural field
        ODIM = 1 - np.identity(self.n_category)

        self.sf_integral = []
        self.sf_integral_norm = []
        for m in range(n_test_seq):  # number of testing condition
            for n in range(self.seq_length):
                # S_tmp = np.squeeze(F_IT[n, m, :, :])  # -> 80x3
                # S_tmp = np.squeeze(data[:, m, n, :])  # marginalize field over the neuron axis
                # todo check normalization!
                S_tmp = np.squeeze(data[:, :, n, m])  # marginalize field over the neuron axis
                for k in range(self.n_category):  # number of training condition
                    # reshape for convenience
                    S_tmp_k = np.reshape(S_tmp[:, k], (S_tmp[:, k].shape[0], 1))

                    # trick to transform the input into a periodic input in the Fourier space and apply the gaussian filter
                    Sff_tmp = np.fft.ifft(np.multiply(np.fft.fft(S_tmp_k, axis=0), self.fftsx), axis=0)

                    # normalize kernel
                    norm_Sf_tmp = self.normalize_fft_kernel(Sff_tmp)

                    Sf_tmp[:, k, n] = norm_Sf_tmp

                    # save mean of integral to help tuning parameters Samp, s_alpha and s_beta
                    if m == k:
                        self.sf_integral.append(np.sum(Sff_tmp))
                        # sf_integral.append(np.amax(Sff_tmp))
                        self.sf_integral_norm.append(np.sum(norm_Sf_tmp))
                        # sf_integral_norm.append(np.amax(norm_Sf_tmp))

            for n in range(1, self.seq_length):  # number of snapshots neurons
                U_tmp = UF[:, :, n - 1]
                # threshold activity
                V_tmp = np.copy(U_tmp)
                V_tmp[V_tmp < 0] = 0

                Vsm_tmp = np.sum(V_tmp, axis=0)  # sum activities for cross-p. inhibition
                Vsm_tmp = np.transpose(np.reshape(Vsm_tmp, (Vsm_tmp.shape[0], 1)))
                # AMARY integral
                CV_tmp = np.zeros(U_tmp.shape)
                for k in range(0, self.n_category):
                    V_tmp_k_fft = np.reshape(np.fft.fft(V_tmp[:, k], axis=0),
                                             (np.fft.fft(V_tmp[:, k], axis=0).shape[0], 1))
                    # compute neural field in the fourrier transform
                    CV_tmp[:, k] = np.squeeze(np.fft.ifft(np.multiply(V_tmp_k_fft, self.fftwx), axis=0))

                # compute leaky integrator
                U_new = (self.tau - 1) * U_tmp + CV_tmp + Sf_tmp[:, :, n - 1] - np.matmul(
                    np.matmul(self.w_inh * np.ones((self.seq_length, 1)), Vsm_tmp), (ODIM)) - self.h

                UF[:, :, n] = U_new / self.tau

            UFA[:, :, :, m] = UF

        UFA = np.maximum(UFA, np.zeros(UFA.shape))

        return UFA

    def predict_dynamic(self, data):
        n_test_seq = np.shape(data)[-1]

        #    compute activity of output neurons (sum over individ. patterns)
        SVDA = np.zeros((data.shape[3], self.seq_length, self.n_category))  # Double-check this later

        for m in range(self.n_category):
            for n in range(self.seq_length):
                SV_tmp = np.squeeze(data[n, m, :, :])
                SV_tmp = np.transpose(np.sum(SV_tmp, axis=0))  # Check that np.sum isn't prod Rank 1 array
                SVDA[:, n, m] = SV_tmp

        # Prededine VD, VDA
        VD = np.zeros((n_test_seq, self.seq_length))
        dyn_resp = np.zeros((n_test_seq, self.seq_length, self.n_category))

        # compute expression neuron dynamics with Euler approximation
        for m in range(self.n_category):
            VD[:, 0] = np.squeeze(np.zeros((n_test_seq, 1)))
            for n in range(1, self.seq_length):
                VD_tmp = VD[:, n - 1]
                VD_new = (self.tau_v - 1) * VD_tmp + np.squeeze(SVDA[:, n, m])
                VD_new = VD_new / self.tau
                VD[:, n] = VD_new
            dyn_resp[:, :, m] = VD

        return dyn_resp

    def plot_integral(self, integral, save_name):
        max_integral = np.amax(integral)
        plt.figure()
        plt.plot(integral)
        plt.plot([80, 80], [0, max_integral])
        plt.plot([160, 160], [0, max_integral])
        plt.savefig(save_name)

    def plot_kernels(self, save_folder=None, title=None):
        # plot interaction kernel
        plt.figure()
        plt.plot(self.wx[0, :])
        plt.title('Interaction kernel')

        fig_title = 'Interaction_kernel.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, fig_title))
        else:
            plt.savefig(fig_title)

        # --------------------------------------------------------------------------
        # plot sf_integral
        sf_integral = np.reshape(np.array(self.sf_integral), -1)

        fig_title = 'sf_integral.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            save_name = os.path.join(save_folder, fig_title)
        else:
            save_name = fig_title
        self.plot_integral(sf_integral, save_name=save_name)

        # --------------------------------------------------------------------------
        # plot sf_integral_norm
        sf_integral_norm = np.reshape(np.array(self.sf_integral_norm), -1)

        fig_title = 'sf_integral_norm.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            save_name = os.path.join(save_folder, fig_title)
        else:
            save_name = fig_title
        self.plot_integral(sf_integral_norm, save_name=save_name)

    def plot_neural_field(self, nn_field, save_folder=None, title=None):
        n_test_seq = nn_field.shape[-1]

        # Section added for plotting F_DFNF_UNF
        UNFc = np.zeros((self.seq_length * self.n_category, self.seq_length * n_test_seq))
        for m in range(n_test_seq):
            for n in range(self.n_category):
                # CAREFUL UFA AXES ARE SWAP WITH RBF_pattern!!!!!!!!!!!!!!!!!!!!
                m_start = m * self.seq_length
                n_start = n * self.seq_length
                UNFc[n_start:n_start + self.seq_length, m_start:m_start + self.seq_length] = np.squeeze(nn_field[:, n, :, m])

        plt.figure()
        im = plt.imshow(np.multiply(UNFc, ((UNFc > 0).astype(int))))
        plt.colorbar(im)
        plt.title('Responses of snapshot neurons')
        plt.xlabel('stimulus frame')
        plt.ylabel('Neuron #')
        print('[testing] neural field response')

        fig_title = 'neural_field_response_test.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, fig_title))
        else:
            plt.savefig(fig_title)

    def plot_dynamic(self, dynamic_resp, save_folder=None, title=None, val=False):
        n_test_seq = np.shape(dynamic_resp)[0]

        # normalize responses for plotting
        dyn_max = np.amax(dynamic_resp)
        dyn_norm = dynamic_resp / dyn_max
        # VDA_norm = VDA / 10.221712329980704  # for reverse

        # plot expression responses
        plt.figure()
        plt.title('Expression Neuron Responses')
        for m in np.arange(n_test_seq):  # n_condition
            plt.subplot(n_test_seq, 1, m + 1)
            for n in range(self.n_category):
                plt.plot(np.transpose(dyn_norm[m, :, n]),
                         color=self.config['colors'][n],
                         linewidth=2)
                if val:
                    plt.ylabel(self.config['val_expression'][n])
                else:
                    plt.ylabel(self.config['train_expression'][n])

                plt.ylim(-0.05, 1.1)

        plt.xlabel('Frames')

        fig_title = 'Expr_Neurons.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, fig_title))
        else:
            plt.savefig(fig_title)