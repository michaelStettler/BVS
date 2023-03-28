import os
import numpy as np

import tensorflow as tf
from datetime import datetime

import matplotlib
from matplotlib import cm

from utils.load_config import load_config
from utils.load_data import load_data
from utils.NRE_optimization.NRE_optimizer import optimize_NRE
from utils.NRE_optimization.NRE_optimizer import estimate_NRE

viridis = cm.get_cmap('viridis', 12)
matplotlib.use('agg')

"""
run: python -m tests.Optimization.t02_optimization_FERG
tensorboard: tensorboard --logdir logs/func


300 epochs, lr = 1, loss = 2.11518, acc = 0.896557
2000 epochs, lr = 1, loss = 2.06291, acc = 0.864654
"""


if __name__ == '__main__':
    profiler = False
    do_plot = False

    # declare parameters
    n_dim = 2
    n_cat = 7
    neutral_cat = None
    n_latent = 10  # == n_lmk
    n_ref = 6  # == n_cat
    lr = 1e-3
    alpha_ref = .06  # strength of the ref cat in the loss function
    batch_size = 512
    n_epochs = 60
    crop_size = 2048
    plot_alpha = 0.1
    use_only_one_cat = None
    early_stopping = False

    # define configuration
    config_file = 'NR_03_FERG_from_LMK_m0001.json'
    config_file = 'NR_03_FERG_from_LMK_w0001.json'
    # config_file = 'NR_03_FERG_alex.json'

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

    # get only the data from the cat of ref
    if use_only_one_cat is not None:
        n_ref = 1
        cat_idx = np.arange(len(train_avatar))[train_avatar == use_only_one_cat]
        train_avatar = train_avatar[cat_idx]
        train_label = train_label[cat_idx]
        train_data = train_data[cat_idx]
        cat_idx = np.arange(len(test_avatar))[test_avatar == use_only_one_cat]
        test_avatar = test_avatar[cat_idx]
        test_data = test_data[cat_idx]
        test_label = test_label[cat_idx]
        print("shape train_data", np.shape(train_data))
        print("shape test_data", np.shape(test_data))
        print("len train_avatar", len(train_avatar))
        print("len test_avatar", len(test_avatar))

    # create labels as [category, identity]
    x_train = train_data
    x_test = test_data
    y_train = np.array([train_label, train_avatar]).T
    y_test = np.array([test_label, test_avatar]).T
    print("shape x_train", np.shape(x_train))
    print("shape y_train", np.shape(y_train))
    print("shape x_test", np.shape(x_test))
    print("shape y_test", np.shape(y_test))

    if crop_size is not None:
        x_train = x_train[:crop_size]
        y_train = y_train[:crop_size]
        print("after crop:")
        print("shape x_train", np.shape(x_train))
        print("shape y_train", np.shape(y_train))


    # transform to tensor
    # init_ref = tf.convert_to_tensor(x_train[[0, 20]] + 0.01, dtype=tf.float64)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    print("shape x_train", x_train.shape)
    print("shape y_train", y_train.shape)

    # set init ref to first neutral
    init_ref = None
    init_ref = []
    for r in range(n_ref):
        # get only the one cat of ref
        if use_only_one_cat is not None:
            r = use_only_one_cat
        ref_pts = x_train[y_train[:, 1] == r]  # take every pts from the ref of interest
        ref_label = y_train[y_train[:, 1] == r]
        neutral_ref_pts = ref_pts[ref_label[:, 0] == 0]  # take only the neutral cat (ref)
        if len(neutral_ref_pts) == 0:
            raise ValueError("No pts found, increase sample size")
        init_ref.append(neutral_ref_pts[0] + np.random.rand(n_latent, n_dim) * 0.01)  # initialize to first neutral pose
    init_ref = tf.convert_to_tensor(init_ref, tf.float32)
    print("init_ref", init_ref.shape)

    if profiler:
        # create logs and tensorboard
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs/func/%s' % stamp
        writer = tf.summary.create_file_writer(logdir)

        tf.summary.trace_on(graph=True, profiler=True)

    # optimize_NRE
    pred, params = optimize_NRE(x_train, y_train, n_cat,
                                batch_size=batch_size,
                                n_ref=n_ref,
                                init_ref=init_ref,
                                lr=lr,
                                alpha_ref=alpha_ref,
                                n_epochs=n_epochs,
                                early_stopping=early_stopping,
                                do_plot=do_plot,
                                plot_alpha=plot_alpha,
                                plot_name="NRE_FERG_optimizer")

    print("finish training")
    print()

    print("training accuracy")
    y_pred = estimate_NRE(x_train, y_train, params,
                          batch_size=batch_size,
                          n_ref=n_ref)
    print()

    print("test accuracy")
    # test accuracy
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
    test_pred = estimate_NRE(x_test, y_test, params,
                             batch_size=batch_size,
                             n_ref=n_ref)



    if profiler:
        with writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir)
