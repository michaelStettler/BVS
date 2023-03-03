import os
import numpy as np

import tensorflow as tf
from datetime import datetime

import matplotlib

from datasets_utils.create_toy_NRE_dataset import generate_data_set
from utils.NRE_optimization.NRE_optimizer import optimize_NRE

matplotlib.use('agg')

np.random.seed(2)

"""
run: python -m tests.Optimization.t01_2d_optimization_algo
tensorboard: tensorboard --logdir logs/func
"""


if __name__ == '__main__':
    profiler = False
    do_plot = True
    shuffle = True

    # declare parameters
    use_ref = True
    n_dim = 2
    n_cat = 4
    neutral_cat = None
    n_points = 10
    n_latent = 3
    n_ref = 3
    n_entry = n_points * n_cat * n_ref
    batch_size = 32
    n_epochs = 20
    lr = 0.1
    alpha_ref = 1
    print(f"{n_entry} entry created!")

    # generate random data
    x_train, y_train = generate_data_set(n_dim, n_cat, n_points,
                                         ref_at_origin=False,
                                         n_latent=n_latent,
                                         n_ref=n_ref,
                                         variance_ratio=5,  # spread the clusters
                                         ref_variance=15,  # spread the ref domain clusters
                                         length_variance=5,  # variance from same cluster on different ref
                                         min_length=6,
                                         max_length=9)  # how far away from ref at minimum
    print("shape x_train", np.shape(x_train))
    print("shape y_train", np.shape(y_train))

    # shuffle
    if shuffle:
        shuffled_idx = np.arange(n_entry)
        np.random.shuffle(shuffled_idx)
        x_train = x_train[shuffled_idx]
        y_train = y_train[shuffled_idx]
    # x_train = [[0, 2], [3, 1], [3, 2], [-1, -1]]
    # y_train = [0, 1, 1, 2]
    # print("shape x_train", np.shape(x_train))


    # plot generated data
    # plot_space(x_train, y_train)

    # transform to tensor
    init_ref = None
    init_ref = []
    for r in range(n_ref):
        ref_pts = x_train[y_train[:, 1] == r]  # take every pts from the ref of interest
        ref_label = y_train[y_train[:, 1] == r]
        neutral_ref_pts = ref_pts[ref_label[:, 0] == 0]  # take only the neutral cat (ref)
        init_ref.append(neutral_ref_pts[0] + np.random.rand(n_latent, n_dim) * 0.01)  # initialize to first neutral pose
    init_ref = tf.convert_to_tensor(init_ref, tf.float32)
    print("init_ref", init_ref.shape)

    # init_ref = tf.convert_to_tensor(x_train[[0, 20]] + 0.01, dtype=tf.float32)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    print("shape x_train", x_train.shape)
    print("shape y_train", y_train.shape)

    if profiler:
        # create logs and tensorboard
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs/func/%s' % stamp
        writer = tf.summary.create_file_writer(logdir)

        tf.summary.trace_on(graph=True, profiler=True)

    # optimize_NRE
    optimize_NRE(x_train, y_train, n_cat, use_ref=use_ref,
                 batch_size=batch_size,
                 n_ref=n_ref,
                 init_ref=init_ref,
                 lr=lr,
                 alpha_ref=alpha_ref,
                 n_epochs=n_epochs,
                 do_plot=do_plot,
                 plot_alpha=0.3,
                 min_plot_axis=25,
                 max_plot_axis=25)

    if profiler:
        with writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir)
