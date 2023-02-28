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
    shuffle = False

    # declare parameters
    use_ref = True
    n_dim = 2
    n_cat = 4
    neutral_cat = None
    n_points = 5
    n_latent = 3
    n_ref = 2
    n_entry = n_points * n_cat * n_ref
    batch_size = 40
    n_epochs = 200
    lr = 0.1
    alpha_ref = 20
    print(f"{n_entry} entry created!")

    # generate random data
    x_train, y_train = generate_data_set(n_dim, n_cat, n_points,
                                         ref_at_origin=False,
                                         n_latent=n_latent,
                                         n_ref=n_ref,
                                         variance_ratio=2,
                                         ref_variance=15,
                                         min_length=3)
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
                 do_plot=do_plot)

    if profiler:
        with writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir)
