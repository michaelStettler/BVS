import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

# matplotlib.use('Agg')  # Added for plotting
plt.style.use('seaborn-paper')  # Added for plotting

def compute_rbf_encoding(input, rbf_center, rbf_sig=0.15):
    """
    Compute the rbf encoding from trained rbf center to new input

    m := n_frames * n_test_seq
    n := n_neurons * n_train_seq

    :param input: (n_feature, m)
    :param rbf_center:(n_feature, n)
    :param rbf_sig:
    :return:
    """

    # f_it = np.zeros((input.shape[1], rbf_center.shape[1]))
    # for m in range(input.shape[1]):
    #     for n in range(rbf_center.shape[1]):
    #         f_it_tmp = input[:, m] - rbf_center[:, n]
    #         f_it_tmp = np.exp(-np.linalg.norm(f_it_tmp, ord=2, axis=2) ** 2 / 2 / rbf_sig ** 2)
    #         f_it[m, n] = f_it_tmp

    # compute difference between rbf center and input for each frame/neurons
    f_it = [[input[:, m] - rbf_center[:, n] for m in range(input.shape[1])] for n in range(rbf_center.shape[1])]
    # apply gaussian activation to each
    f_it = np.exp(-np.linalg.norm(f_it, ord=2, axis=2)**2 / 2 / rbf_sig**2)

    return f_it


def reshape_rbf(rbf, m_frame, m_test, n_neuron, n_train):
    """
    this function reshape the matrix from (n, m) dimension to (m_frame, m_test, n_neuron, n_train)
    with

    m := m_frame * m_test_seq
    n := n_neuron * n_train_seq

    Note: usually n_frame = n_neuron

    :param rbf:
    :return:
    """

    # declare new array
    F_IT_rbf = np.zeros((m_frame, m_test, n_neuron, n_train))

    # reshape array
    for n in range(n_train):
        for m in range(m_test):
            n_start = n * n_neuron
            m_start = m * m_frame
            F_IT_rbf[:, m, :, n] = rbf[n_start:n_start+n_neuron, m_start:m_start+m_frame]

    return F_IT_rbf


def normalize_fft_kernel(s, A, alpha, beta, epsilon=.1):
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

    norm = ((1 - beta) / alpha) + (beta / (epsilon + np.sum(s)))
    s_norm = np.squeeze(A * s * norm)

    return s_norm


def remove_neutral_frames(F_IT, config):
    neutral_index = config['neutral_frames_idx']

    for i in neutral_index:
        F_IT[i[0]:i[1]] = 0
        F_IT[:, i[0]:i[1]] = 0

    return F_IT


def reverse_sequence(seq, n_seq, seq_length):
    rev_seq = np.zeros(seq.shape)
    for s in range(n_seq):
        start = s * seq_length
        chopped = seq[:, start:start+seq_length]
        print("shape chopped", np.shape(chopped))
        flip = np.flip(seq[:, start:start+seq_length], axis=1)

        rev_seq[:, start:start+seq_length] = flip

    return rev_seq


def plot_integral(integral, save_name):
    max_integral = np.amax(integral)
    plt.figure()
    plt.plot(integral)
    plt.plot([80, 80], [0, max_integral])
    plt.plot([160, 160], [0, max_integral])
    plt.savefig(save_name)


def compute_expression_neurons(F_IT, config, do_plot=1):
    print("[dface] Input shape F_IT", np.shape(F_IT))

    # remove neutral frames so the neural field is acting only on the expression sequence
    if config.get('remove_neutral_frames') is not None:
        F_IT = remove_neutral_frames(F_IT, config)

    seq_length = config['batch_size']
    n_test_seq = F_IT.shape[1] // seq_length
    n_train_seq = F_IT.shape[0] // seq_length
    print("n_train_seq", n_train_seq)
    print("n_test_seq", n_test_seq)

    # reshape R_IT_rbfc from (n, m) to (m_frame, m_test, n_neuron, n_train)
    F_IT = reshape_rbf(F_IT, seq_length, n_test_seq, seq_length, n_train_seq)
    print("[dface] re-shape F_IT", np.shape(F_IT))

    # Initialize Neural Field
    """
    Parameters to modify to fine tune the neural field, in order of importance
    Aker: Amplitude of the kernel, you should try to make your field activated on the diagonal, the more you will add, 
    the more the diagonal will start to lurch
    Bker: offset of the amplitude kernel, it allows to make the kernel inhibitroy outside its peak
    dker: tune the speed of the traveling pulse, it will correct the lurching, to test sequence selectivity, 
    set the parameter to negatif and control that your fiedl is well sequence selective
    
    winh: this how strong the pattern will inhibit the others
    
    Samp/s_alpha/s_beta: amplitude and normalization of the fourier kernel, in our case an amplitude of the kernel 
    around 75 was good, the goal is to try to flatten the kernel but keeping it's normalization amplitude somehow 
    similar across the fields
    """
    Aker = 1.15  # 1.1 kernel amplitude
    Bker = -0.5  # kernel offset
    dker = 3.1  # asymmetric shift

    winh = 0.5  # cross-pattern inhibition

    Samp = 25  # amplitude factor of normalization
    s_alpha = 40  # minimum normalization factor
    s_beta = 0.4  # blending parameters

    """
    Following parameters were not touched to tune the neural field
    """
    sigker = 2.5  # interaction kernel
    h = 1  # resting level
    tau = 5  # time constant
    sigs = 1.8  # gaussian lurring of input distribution

    # ----------------------------------------------------------
    # ---------------------- AMARY FIELD -----------------------
    # ----------------------------------------------------------

    # create the interaction kernel
    xs = np.arange(0, seq_length)
    xsft = math.floor(seq_length / 2)
    hsft = 0 * xs
    hsft[xsft] = 1
    hsft = hsft.reshape(1, hsft.shape[0])

    # build interaction kernel
    wx = Aker * np.exp(-(xs - xsft - dker) ** 2 / 2 / sigker ** 2) + Bker
    wx = wx.reshape(1, wx.shape[0])
    wx = np.round(wx, 4)
    wx = np.fft.ifft(np.multiply(np.fft.fft(wx), np.conj(np.fft.fft(hsft))))  # trick to center the kernel
    fftwx = np.fft.fft(np.transpose(wx), axis=0)

    #    create input smoothing kernel
    sx = np.exp(-(xs - xsft) ** 2 / 2 / sigs ** 2)
    sx = np.fft.ifft(np.multiply(np.fft.fft(sx), np.conj(np.fft.fft(hsft))))  # trick to center the kernel

    # fourrier transform of the input kernel
    fftsx = np.fft.fft(np.transpose(sx), axis=0)

    #    initialize neural field and define UF, UFA, Sf_tmp
    Uf0 = - np.ones((seq_length, n_train_seq)) * h

    UF = np.zeros((seq_length, n_train_seq, seq_length))
    UFA = np.zeros((seq_length, n_train_seq, seq_length, n_test_seq))
    Sf_tmp = np.zeros((seq_length, n_train_seq, seq_length))

    #    iterate neural field
    ODIM = 1 - np.identity(n_train_seq)

    sf_integral = []
    sf_integral_norm = []
    for m in range(n_test_seq):  # number of testing condition
        UF[:, :, 0] = Uf0  # initialization for time 1
        for n in range(seq_length):
            # S_tmp = np.squeeze(F_IT[n, m, :, :])  # -> 80x3
            S_tmp = np.squeeze(F_IT[:, m, n, :])  # marginalize field over the neuron axis
            for k in range(n_train_seq):  # number of training condition
                # reshape for convenience
                S_tmp_k = np.reshape(S_tmp[:, k], (S_tmp[:, k].shape[0], 1))

                # trick to transform the input into a periodic input in the Fourier space and apply the gaussian filter
                Sff_tmp = np.fft.ifft(np.multiply(np.fft.fft(S_tmp_k, axis=0), fftsx), axis=0)

                # normalize kernel
                norm_Sf_tmp = normalize_fft_kernel(Sff_tmp, Samp, s_alpha, s_beta)

                Sf_tmp[:, k, n] = norm_Sf_tmp

                # save mean of integral to help tuning parameters Samp, s_alpha and s_beta
                if m == k:
                    sf_integral.append(np.sum(Sff_tmp))
                    # sf_integral.append(np.amax(Sff_tmp))
                    sf_integral_norm.append(np.sum(norm_Sf_tmp))
                    #sf_integral_norm.append(np.amax(norm_Sf_tmp))

        for n in range(1, seq_length):  # number of snapshots neurons
            U_tmp = UF[:, :, n - 1]
            # threshold activity
            V_tmp = np.copy(U_tmp)
            V_tmp[V_tmp < 0] = 0

            Vsm_tmp = np.sum(V_tmp, axis=0)  # sum activities for cross-p. inhibition
            Vsm_tmp = np.transpose(np.reshape(Vsm_tmp, (Vsm_tmp.shape[0], 1)))
            # AMARY integral
            CV_tmp = np.zeros(U_tmp.shape)
            for k in range(0, n_train_seq):
                V_tmp_k_fft = np.reshape(np.fft.fft(V_tmp[:, k], axis=0), (np.fft.fft(V_tmp[:, k], axis=0).shape[0], 1))
                # compute neural field in the fourrier transform
                CV_tmp[:, k] = np.squeeze(np.fft.ifft(np.multiply(V_tmp_k_fft, fftwx), axis=0))

            # compute leaky integrator
            U_new = (tau - 1) * U_tmp + CV_tmp + Sf_tmp[:, :, n - 1] - np.matmul(
                np.matmul(winh * np.ones((seq_length, 1)), Vsm_tmp), (ODIM)) - h

            UF[:, :, n] = U_new / tau

        UFA[:, :, :, m] = UF

    YFA = np.maximum(UFA, np.zeros(UFA.shape))

    #    compute activity of output neurons (sum over individ. patterns)
    tau_v = 4  # time constant (read out neurons)

    SVDA = np.zeros((YFA.shape[3], seq_length, n_train_seq))  # Double-check this later

    for m in range(n_train_seq):
        for n in range(seq_length):
            SV_tmp = np.squeeze(YFA[n, m, :, :])
            SV_tmp = np.transpose(np.sum(SV_tmp, axis=0))  # Check that np.sum isn't prod Rank 1 array
            SVDA[:, n, m] = SV_tmp


    # Prededine VD, VDA
    VD = np.zeros((n_test_seq, seq_length))  # Double-check this later
    VDA = np.zeros((n_test_seq, seq_length, n_train_seq))  # Double-check later

    # compute expression neuron dynamics with Euler approximation
    for m in range(n_train_seq):
        VD[:, 0] = np.squeeze(np.zeros((n_test_seq, 1)))
        for n in range(1, seq_length):
            VD_tmp = VD[:, n - 1]
            VD_new = (tau_v - 1) * VD_tmp + np.squeeze(SVDA[:, n, m])
            VD_new = VD_new / tau
            VD[:, n] = VD_new
        VDA[:, :, m] = VD

    # ----------------------------------------------------------
    # ----------------------     PLOT    -----------------------
    # ----------------------------------------------------------

    with open('expression_neurons.pkl', 'wb') as f:
        pickle.dump(VDA, f)

    if do_plot > 0:
        # plot interaction kernel
        plt.figure()
        plt.plot(wx[0, :])
        plt.title('Interaction kernel')
        plt.savefig('Interaction_kernel.png')

        # plot sf_integral
        sf_integral = np.reshape(np.array(sf_integral), -1)
        plot_integral(sf_integral, save_name='sf_integral')

        # plot sf_integral_norm
        sf_integral_norm = np.reshape(np.array(sf_integral_norm), -1)
        plot_integral(sf_integral_norm, save_name='sf_integral_norm')

        # Section added for plotting F_DFNF_UNF
        print("UFA size", UFA.shape)
        UNFc = np.zeros((seq_length * n_train_seq, seq_length * n_test_seq))
        for m in range(n_test_seq):
            for n in range(n_train_seq):
                # CAREFUL UFA AXES ARE SWAP WITH RBF_patch_pattern!!!!!!!!!!!!!!!!!!!!
                m_start = m * seq_length
                n_start = n * seq_length
                UNFc[n_start:n_start + seq_length, m_start:m_start + seq_length] = np.squeeze(UFA[:, n, :, m])

        plt.figure()
        im = plt.imshow(np.multiply(UNFc, ((UNFc > 0).astype(int))))
        plt.colorbar(im)
        plt.title('Responses of snapshot neurons')
        plt.xlabel('stimulus frame')
        plt.ylabel('Neuron #')
        print('[testing] neural field response')
        plt.savefig('neural_field_response_test')

        # plot expression responses
        VDA_max = np.amax(VDA)
        print("VDA Max", VDA_max)
        VDA_norm = VDA / VDA_max
        # VDA_norm = VDA / 10.221712329980704  # for reverse
        plt.figure()
        plt.title('Expression Neuron Responses')
        print("shape VDA", np.shape(VDA))
        for m in np.arange(n_test_seq):  # n_condition
            plt.subplot(n_test_seq, 1, m + 1)
            for n in range(n_train_seq):
                plt.plot(np.transpose(VDA_norm[m, :, n]),
                         color=config['colors'][n],
                         linewidth=2)
                plt.ylabel(config['condition'][m])
                plt.ylim(-0.05, 1.1)

        plt.xlabel('Frames')
        plt.savefig('Expr Neurons.png')

    return UFA, VDA


def compute_snapshot_neurons(config, do_plot=1, save=1, do_reverse=0):
    # Loading saved parameters
    with open('Seq_train_pars.pkl', 'rb') as f:
        RBFct, firing_thr_rbf, rbf_sig = pickle.load(f)
    print("[SNAP] shape RBFct", np.shape(RBFct))

    #TSDATARR,n_condition = pickle.load(open('Test_seq_features.pkl', 'rb'))
    with open('Test_seq_features.pkl', 'rb') as f:
        TSDATARR, n_condition = pickle.load(f)
    ZDMAT = np.transpose(TSDATARR)
    szZDMAT = ZDMAT.shape
    print("[SNAP] shape ZDMAT", szZDMAT)

    n_frames = ZDMAT.shape[1]
    n_neuron = RBFct.shape[1]
    seq_length = config['batch_size']
    n_test_seq = ZDMAT.shape[1] // seq_length
    n_train_seq = RBFct.shape[1] // seq_length

    if do_reverse:
        print("[SNAP] Reverse sequence")
        ZDMAT = reverse_sequence(ZDMAT, n_test_seq, seq_length)

    ## RBF_patch_pattern ENCODING
    # compute response of input to RBF_patch_pattern
    F_IT_rbfc = compute_rbf_encoding(ZDMAT, RBFct, rbf_sig=rbf_sig)

    #      response statistics
    sig_fire = np.mean(F_IT_rbfc > firing_thr_rbf)
    print('[SNAP] RBF_patch_pattern neurons fire on average for ' + str(sig_fire * 100) + ' % of the training stimuli.')

    #Next section added for plotting
    if do_plot > 0:
        plt.figure()
        im = plt.imshow(F_IT_rbfc)
        plt.colorbar(im)
        for m in range(1, n_test_seq):
            # draw vertical line
            plt.plot([m*seq_length, m*seq_length], [0, n_neuron - 1], color='r')
        for n in range(1, n_train_seq):
            # draw hori line
            plt.plot([0, n_frames - 1], [n*seq_length, n*seq_length], color='r')
        plt.title('IT neuron (RBF_patch_pattern) responses')
        plt.xlabel('# of stimulus')
        plt.ylabel('Neuron #')
        plt.savefig('RBF_testresponse.png')

    if save:
        with open('RBF_snapshot_neurons.pkl', 'wb') as f:
            pickle.dump(F_IT_rbfc, f)

    return F_IT_rbfc


if __name__ == '__main__':
    """
    Example of use
    """
    import json
    import os

    config_path = 'configs/example_base_config'
    config_name = 'example_base_reproduce_ICANN_cat.json'
    # config_name = 'example_base_reproduce_ICANN_expressivity.json'
    #config_name = 'example_base_monkey_test_reduced_dataset_.json'

    config_file_path = os.path.join(config_path, config_name)
    print("config_file_path", config_file_path)

    #Load example_base_config file
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    # compute snapchot neurons repsonse
    snap = compute_snapshot_neurons(config)

    # compute expression neurons
    compute_expression_neurons(snap, config)
