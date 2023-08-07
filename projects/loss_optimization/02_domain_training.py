import os
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from datetime import datetime

import matplotlib
from matplotlib import cm

from utils.load_config import load_config
from utils.load_data import load_data
from utils.NRE_optimization.NRE_optimizer import fit_NRE
from utils.NRE_optimization.NRE_optimizer import transfer_domains

viridis = cm.get_cmap('viridis', 12)
matplotlib.use('agg')

"""
run: python -m projects.loss_optimization.01_subset_training
"""

def get_domain_subset(x_train, y_train, x_test, y_test, domain):
    """
    Selects images from domain from training set
    """
    # select all samples from domain
    train_idx = list(np.where(y_train[:, 1] == domain)[0])

    # select first neutral from each domain
    for d in range(6):
        if d == domain:
            continue
        first_d_idx = np.where(np.logical_and(y_train[:, 1] == d, y_train[:, 0] == 0))[0][0]
        train_idx.append(first_d_idx)

    x_train = x_train[train_idx, :, :]
    y_train = y_train[train_idx, :]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # set seed
    np.random.seed(42)
    do_plot = False
    # save_path = 'D:/Dataset/FERG_DB_256/loss_optimization'
    save_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results'

    # declare parameters
    n_dim = 2
    n_cat = 7
    n_latent = 10  # == n_lmk
    n_ref = 6  # == n_cat
    lr = 1e-3
    n_epochs = 250
    lr_decay = [100]
    early_stopping = False

    alpha_ref = 1

    # define configuration
    # config_file = 'NR_03_FERG_from_LMK_m0001.json'
    # config_file = 'NR_03_FERG_from_LMK_w0001.json'
    config_file = 'NR_03_FERG_alex.json'

    # load config
    # config = load_config(config_file, path='/Users/michaelstettler/PycharmProjects/BVS/BVS/configs/norm_reference')
    # config = load_config(config_file, path='D:/PycharmProjects/BVS/configs/norm_reference')
    config = load_config(config_file, path=r'C:\Users\Alex\Documents\Uni\NRE\BVS\configs\norm_reference')
    print("-- Config loaded --")
    print()

    # Load data
    train_data = load_data(config, get_raw=True, get_only_label=True)
    train_label = train_data[1]
    test_data = load_data(config, train=False, get_raw=True, get_only_label=True)
    test_label = test_data[1]
    print("shape train_data[0]", np.shape(train_data[0]))
    print("shape test_data[0]", np.shape(test_data[0]))

    # load lmk pos
    train_data = np.load(config['train_lmk_pos'])
    test_data = np.load(config['test_lmk_pos'])
    print("shape train_data", np.shape(train_data))
    print("shape test_data", np.shape(test_data))

    # load avatar types
    train_avatar = np.load(config['train_avatar_type'])
    test_avatar = np.load(config['test_avatar_type'])
    print("shape train_avatar", np.shape(train_avatar))
    print("shape test_avatar", np.shape(test_avatar))
    print("-- Data loaded --")
    print()

    # create labels as [category, identity]
    x_train = train_data
    x_test = test_data
    y_train = np.array([train_label, train_avatar]).T
    y_test = np.array([test_label, test_avatar]).T
    print("shape x_train", np.shape(x_train))
    print("shape y_train", np.shape(y_train))
    print("shape x_test", np.shape(x_test))
    print("shape y_test", np.shape(y_test))

    # # test only on non-neutrals
    # non_neutrals = np.where(y_test[:, 0] != 0)[0]
    # x_test = x_test[non_neutrals]
    # y_test = y_test[non_neutrals]

    out = {}
    for domain in range(6):
        print('----------------------------------------------------------')
        print('Domain:', domain)
        x_train_dom, y_train_dom, x_test_dom, y_test_dom = get_domain_subset(x_train, y_train, x_test, y_test, domain)

        # transform to tensor
        # init_ref = tf.convert_to_tensor(x_train[[0, 20]] + 0.01, dtype=tf.float64)
        x_train_dom = tf.convert_to_tensor(x_train_dom, dtype=tf.float32)
        y_train_dom = tf.convert_to_tensor(y_train_dom, dtype=tf.int32)
        # print(y_sub)
        print("shape x_train_dom", x_train_dom.shape)
        print("shape y_train_dom", y_train_dom.shape)
        print("shape x_test_dom", x_test_dom.shape)
        print("shape y_test_dom", y_test_dom.shape)

        # set init ref to first neutral
        # in this case, only necessary for the domain that is being used to train
        init_ref = []
        for r in range(n_ref):
            ref_pts = x_train_dom[y_train_dom[:, 1] == r]  # take every pts from the ref of interest
            ref_label = y_train_dom[y_train_dom[:, 1] == r]
            neutral_ref_pts = ref_pts[ref_label[:, 0] == 0]  # take only the neutral cat (ref)
            neutral_ref_pts = ref_pts[ref_label[:, 0] == 0]  # take only the neutral cat (ref)
            if len(neutral_ref_pts) == 0:
                raise ValueError("No pts found, increase sample size")
            init_ref.append(neutral_ref_pts[0] + np.random.rand(n_latent, n_dim) * 0.01)  # initialize to first neutral pose
        init_ref = tf.convert_to_tensor(init_ref, tf.float32)
        print("init_ref", init_ref.shape)

        # optimize_NRE
        batch_size = len(x_train_dom)
        train_accuracies = {}
        test_accuracies = {}
        pred, params, metrics = fit_NRE(x_train_dom, y_train_dom, n_cat,
                                        x_test=x_test_dom,
                                        y_test=y_test_dom,
                                        batch_size=batch_size,
                                        n_ref=n_ref,
                                        init_ref=init_ref,
                                        lr=lr,
                                        alpha_ref=alpha_ref,
                                        n_epochs=n_epochs,
                                        early_stopping=early_stopping,
                                        trained_domain=domain)


        print("finish training")
        print(f"best_accuracy: {metrics['best_acc']}")
        print()

        # append results
        out[domain] = metrics

    # save results
    save_name = 'single_domain_training'
    with open(os.path.join(save_path, save_name), 'wb') as f:
        pickle.dump(out, f)

