import numpy as np

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
        self.seq_length = config['batch_size']

        # declare neural field tuning parameters
        self.Aker = config["Amari_Aker"]                # kernel amplitude
        self.Bker = config["Amari_Bker"]                # kernel offset
        self.Dker = config["Amari_Dker"]                # asymmetric shift
        self.w_inh = config["Amari_w_inh"]              # cross-pattern inhibition
        self.sigs = config["Amari_sigs"]                # gaussian luring of input distribution
        self.h = config["Amari_h"]                      # resting level
        self.tau = config["Amari_tau"]                  # time constant
        self.s_amp = config["Amari_nker_s_amp"]         # amplitude factor of normalization
        self.s_alpha = config["Amari_nker_s_alpha"]     # minimum normalization factor
        self.s_beta = config["Amari_nker_s_beta"]       # blending parameters
        self.sig_ker = config["Amari_ker_sig"]          # interaction kernel

        # declare neural field parameters
        self.xs = np.arange(0, self.seq_length)
        self.xsft = np.floor(self.seq_length / 2)
        self.hsft = None                                # interaction kernel
        self.fftwx = None                               # interaction kernel (Fourier domain)
        self.sx = None                                  # smoothing kernel
        self.fftsx = None                               # smoothing kernel (Fourier domain)

    def _build_inter_kernel(self):
        """
        create the interaction kernel

        :return:
        """
        # create kernel
        hsft = 0 * self.xs
        hsft[self.xsft] = 1
        self.hsft = hsft.reshape(1, hsft.shape[0])

        # build interaction kernel
        wx = self.Aker * np.exp(-(self.xs - self.xsft - self.Dker) ** 2 / 2 / self.sig_ker ** 2) + self.Bker
        wx = wx.reshape(1, wx.shape[0])
        wx = np.round(wx, 4)
        wx = np.fft.ifft(np.multiply(np.fft.fft(wx), np.conj(np.fft.fft(hsft))))  # trick to center the kernel
        self.fftwx = np.fft.fft(np.transpose(wx), axis=0)

    def _build_smooth_input_kernel(self):
        # create input smoothing kernel
        self.sx = np.exp(-(self.xs - self.xsft) ** 2 / 2 / self.sigs ** 2)
        self.sx = np.fft.ifft(np.multiply(np.fft.fft(self.sx), np.conj(np.fft.fft(self.hsft))))  # trick to center the kernel

        # fourrier transform of the input kernel
        self.fftsx = np.fft.fft(np.transpose(self.sx), axis=0)
