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

viridis = cm.get_cmap('viridis', 12)
matplotlib.use('agg')

"""
run: python -m projects.loss_optimization.01_subset_training
"""

def get_groups(y):
    unique_labels = np.unique(y[:, 0])
    groups = {}
    for cat in unique_labels:
        groups[cat] = list(np.where(y[:, 0] == cat)[0])
    return groups

def add_samples(x_train, y_train, groups, x_sub=None, y_sub=None, n_samples=1):
    if x_sub is None:
        x_sub = np.zeros((0, 10, 2))
        y_sub = np.zeros((0, 2))
    # loop over unique categories (7)
    for cat in groups.keys():
        # take random indices from group
        idx = np.random.choice(groups[cat], size=n_samples, replace=False)
        # remove index from pool
        for id in idx:
            groups[cat].remove(id)
        # add index to the training set
        x_sub = np.concatenate((x_sub, x_train[idx, :, :]), 0)
        y_sub = np.concatenate((y_sub, y_train[idx, :]), 0)
    return groups, x_sub, y_sub


if __name__ == '__main__':
    do_plot = False
    # save_path = 'D:/Dataset/FERG_DB_256/loss_optimization'
    save_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results'

    # declare parameters
    n_dim = 2
    n_cat = 7
    n_latent = 10  # == n_lmk
    n_ref = 6  # == n_cat
    lr = 1e-5
    n_epochs = 400
    early_stopping = False

    alpha_ref = [0.06]

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

    # Get training indices separated by group
    groups = get_groups(y_train)

    # Initialize training subsets
    x_sub, y_sub = None, None

    # Number of images to add on each epoch
    subset_increaser = np.concatenate((np.array([1]), 2 ** np.arange(12)))

    # Loop to add samples to the training set
    for n_new in subset_increaser:
        groups, x_sub, y_sub = add_samples(x_train=x_train, y_train=y_train,
                                           groups=groups, x_sub=x_sub, y_sub=y_sub, n_samples=n_new)
        print(x_sub.shape)
        print(y_sub.shape)
        n_sub = x_sub.shape[0]

        # transform to tensor
        # init_ref = tf.convert_to_tensor(x_train[[0, 20]] + 0.01, dtype=tf.float64)
        x_sub = tf.convert_to_tensor(x_sub, dtype=tf.float32)
        y_sub = tf.convert_to_tensor(y_sub, dtype=tf.int32)
        print("shape x_sub", x_sub.shape)
        print("shape y_sub", y_sub.shape)


        # set init ref to first neutral
        init_ref = []
        for r in range(n_ref):
            ref_pts = x_sub[y_sub[:, 1] == r]  # take every pts from the ref of interest
            ref_label = y_sub[y_sub[:, 1] == r]
            neutral_ref_pts = ref_pts[ref_label[:, 0] == 0]  # take only the neutral cat (ref)
            if len(neutral_ref_pts) == 0:
                raise ValueError("No pts found, increase sample size")
            init_ref.append(neutral_ref_pts[0] + np.random.rand(n_latent, n_dim) * 0.01)  # initialize to first neutral pose
        init_ref = tf.convert_to_tensor(init_ref, tf.float32)
        print("init_ref", init_ref.shape)

        # optimize_NRE
        batch_size = len(x_sub)
        train_accuracies = {}
        test_accuracies = {}
        pred, params, metrics = fit_NRE(x_sub, y_sub, n_cat,
                                        x_test=x_test,
                                        y_test=y_test,
                                        batch_size=batch_size,
                                        n_ref=n_ref,
                                        init_ref=init_ref,
                                        lr=lr,
                                        alpha_ref=alpha_ref,
                                        n_epochs=n_epochs,
                                        early_stopping=early_stopping)

        print("finish training")
        print(f"best_accuracy: {metrics['best_acc']}")
        print()

        # append results
        train_accuracies[n_sub] = metrics['train_accuracies']
        test_accuracies[n_sub] = metrics['test_accuracies']

    # save results

    with open(os.path.join(save_path, 'train_n_accuracy'), 'wb') as f:
        pickle.dump(train_accuracies, f)

    with open(os.path.join(save_path, 'test_n_accuracy'), 'wb') as f:
        pickle.dump(test_accuracies, f)