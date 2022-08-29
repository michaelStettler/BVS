from sklearn.decomposition import PCA

do_plot = 1  # Added for plotting
do_nice = 0  # Added for plotting
import matplotlib
import matplotlib.pyplot as plt  # Added for plotting
import numpy as np
import math
import pickle
import json
import os

# matplotlib.use('Agg')  # Added for plotting
plt.style.use('seaborn-paper')  # Added for plotting


def train_model_seq(config):
    import scipy.io

    TRDATARR = pickle.load(open('Train_seq_features.pkl', 'rb')) 
    
    ZDMAT = np.transpose(TRDATARR)
    print("shape ZDMAT", ZDMAT.shape)
    # print(ZDMAT)

    n_train_category = config['n_train_category']
    
    
    szZDMAT = ZDMAT.shape
    Nt = szZDMAT[1]/n_train_category  # =3, Length of TRP got from original code OR len(TRP)
    #split data array in prototypes and compute reference pattern (mean of first frames)
    Nptt = n_train_category  # =3, Length of TRP got from original code OR len(TRP)
     
    ##RBF_pattern ENCODING
    firing_thr_rbf = 0.15

    #      train recognition units
    RBFct = ZDMAT

    #      compute responses of RBF_pattern units for all training stimuli
    rbf_sig = 2.5  # 0.1; #0.05; 0.15;0.2;0.53;0.6;
    
    n_train_cond = (RBFct.shape[1])/Nt

    szRBFct = RBFct.shape
    F_IT_rbfc = np.zeros((szRBFct[1],szRBFct[1]))
    for n in range(0,szRBFct[1]):
        for m in range(0,szRBFct[1]):
            x_tmp = ZDMAT[:, n] - RBFct[:, m]
            x_tmp = math.exp(-np.linalg.norm(x_tmp,2)**2 / 2 / rbf_sig**2)  # norm(var,2)= 2-norm = matlab's norm
            F_IT_rbfc[m, n] = x_tmp
           

    #      reorder in response array
    F_IT_rbf = np.zeros((int(Nt),3,int(Nt),3))
    for ntp in range(int(Nptt)):
        for m in range(int(Nptt)):
            mindex = np.arange(int(Nt)) + m*int(Nt)
            pindex = np.arange(int(Nt)) + ntp*int(Nt)
            mlen = len(mindex)  # not really needed
            plen = len(pindex)  # not really needed
            F_IT_rbf[:, m, :, ntp] = F_IT_rbfc[mindex[0]:mindex[mlen-1]+1, pindex[0]:pindex[plen-1]+1]
            #print(F_IT_rbfc[mindex[0]:mindex[mlen-1]+1, pindex[0]:pindex[plen-1]+1].shape)

    #      response statistics
    print(F_IT_rbfc)
    sig_fire = F_IT_rbfc > firing_thr_rbf
    sig_fire = np.mean(sig_fire)
    print('RBF_pattern neurons fire on average for ' + str(sig_fire * 100) + ' % of the training stimuli.')
    
    # Plot RBF_pattern section (until##)
    if do_plot > 0:
        plt.clf
        plt.cla
        im = plt.imshow(F_IT_rbfc)
        plt.colorbar(im)
        plt.title('RBF_responses')
        plt.savefig('RBF_trainresponses.png')    
    
    # Save training parameters
    print("Saving training parameters")
    with open('Seq_train_pars.pkl', 'wb') as f: 
        pickle.dump([RBFct, firing_thr_rbf, rbf_sig], f)
    
    return F_IT_rbf

##    #      save training parameters -- change from MATLAB syntax
##    sv_file = [sv_file];
##    disp(['Saving file ', sv_file, '.']);
##    # save(sv_file, 'sel_index', 'PRM', 'DATmn', 'MVact',  'DPM',  'NVEC', ...
##    #               'nupar', 'neur_type', 'RBFct', 'rbf_sig');
##              
##    #save(sv_file, 'sel_index', 'PRM', 'DATmn', 'tsref', 'npcfeat', ...
##         'nupar', 'ZCnn', 'RBFct', 'firing_thr_rbf', 'rbf_sig');


if __name__ == '__main__':
    config_path = 'configs/example_base/example_base_config'
    config_name = 'example_base_reproduce_ICANN_cat.json'

    config_file_path = os.path.join(config_path, config_name)
    print("config_file_path", config_file_path)

    #Load example_base_config file
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    train_model_seq(config)

