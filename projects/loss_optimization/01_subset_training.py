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

def get_first_samples(x_train, y_train):
    x_sub = np.zeros((0, 10, 2))
    y_sub = np.zeros((0, 2))

    # partition data by category
    unique_labels = np.unique(y_train[:, 0])
    groups = {}
    for cat in unique_labels:
        groups[cat] = list(np.where(y_train[:, 0] == cat)[0])

    # loop over non-neutral categories (6) and add a random sample from each category
    for cat in groups.keys():
        if cat == 0:
            # skip neutral expression for now
            continue
        # take random index from group
        id = np.random.choice(groups[cat], size=1, replace=False)
        # remove index from pool
        groups[cat].remove(id)
        # add index to the training set
        x_sub = np.concatenate((x_sub, x_train[id, :, :]), 0)
        y_sub = np.concatenate((y_sub, y_train[id, :]), 0)

    # loop over avatars and take a random neutral sample from each avatar
    for avatar in range(6):
        idx = list(np.where((y_train == (0, avatar)).all(axis=1))[0])
        id = np.random.choice(idx, size=1, replace=False)
        # remove index from pool
        groups[0].remove(id)
        # add index to the training set
        x_sub = np.concatenate((x_sub, x_train[id, :, :]), 0)
        y_sub = np.concatenate((y_sub, y_train[id, :]), 0)

    # undo partition into groups to sample randomly from now on
    remaining_indices = []
    for cat in groups.keys():
        remaining_indices.extend(groups[cat])
    return remaining_indices, x_sub, y_sub

def add_samples(x_train, y_train, remaining_indices, x_sub=None, y_sub=None, n_samples=1):
    new_idx = np.random.choice(remaining_indices, size=n_samples, replace=False)
    for id in new_idx:
        remaining_indices.remove(id)

    x_sub = np.concatenate((x_sub, x_train[new_idx, :, :]), 0)
    y_sub = np.concatenate((y_sub, y_train[new_idx, :]), 0)
    return remaining_indices, x_sub, y_sub


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
    lr_decay = [75, 200, 300]
    early_stopping = False

    alpha_ref = 0.06

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

    # Initialize training subsets
    remaining_indices, x_sub, y_sub = get_first_samples(x_train, y_train)

    # Number of images to add on each epoch
    subset_increaser = np.concatenate((np.array([1]), 2 ** np.arange(15)))

    # Loop to add samples to the training set
    for n_new in subset_increaser:
        remaining_indices, x_sub, y_sub = add_samples(x_train=x_train, y_train=y_train,
                                                      remaining_indices=remaining_indices, x_sub=x_sub, y_sub=y_sub,
                                                      n_samples=n_new)
        n_sub = x_sub.shape[0]
        print('number of images:', n_sub)

        # transform to tensor
        # init_ref = tf.convert_to_tensor(x_train[[0, 20]] + 0.01, dtype=tf.float64)
        x_sub = tf.convert_to_tensor(x_sub, dtype=tf.float32)
        y_sub = tf.convert_to_tensor(y_sub, dtype=tf.int32)
        print(y_sub)
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