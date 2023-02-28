import os
import numpy as np

import cv2
import tensorflow as tf
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.load_config import load_config
from utils.load_data import load_data

viridis = cm.get_cmap('viridis', 12)
matplotlib.use('agg')

np.random.seed(2)

"""
run: python -m tests.Optimization.t02_optimization_FERG
tensorboard: tensorboard --logdir logs/func
"""


def batch(x, y, n=32):
    l = len(x)
    for ndx in range(0, l, n):
        yield x[ndx:min(ndx + n, l)], y[ndx:min(ndx + n, l)]


def compute_tun_vectors(x, y, n_cat):
    n_feat_maps = np.shape(x)[1]
    n_dim = np.shape(x)[-1]

    # if no data comes in
    if y.ndim == 0:
        # careful because as y is empty, then x shape changes to ndim = 2
        return tf.zeros((n_cat, np.shape(x)[0], n_dim), dtype=tf.float64)

    tun_vectors = []
    # for each expression
    for cat in range(n_cat):
        # construct mask to get slice of tensor x per category (in numpy x[y == cat])
        bool_mask = tf.equal(y, cat)
        x_cat = tf.gather(x, tf.squeeze(tf.where(bool_mask)))

        # only if there's sample of this category in the batch
        if x_cat.shape[0] != 0:
            # if only one point in the category
            if len(x_cat.shape) == 2:
                x_cat = tf.expand_dims(x_cat, axis=0)

            # declare tuning vect per category
            # v_cat = tf.zeros((n_feat_maps, n_dim))
            v_cat = []
            for f in range(n_feat_maps):
                # svd results not consistent between torch and tf
                s, u, vh = tf.linalg.svd(x_cat[:, f], full_matrices=False)
                # print("shape u, s, vh", u.shape, s.shape, vh.shape)
                # print(vh)

                # Orient tuning vectors properly
                vh = tf.transpose(vh)
                x_direction = tf.gather(x_cat[:, f], tf.math.argmax(tf.norm(x_cat[:, f], axis=-1)))[0]
                y_direction = tf.gather(x_cat[:, f], tf.math.argmax(tf.norm(x_cat[:, f], axis=-1)))[1]
                x_direction = x_direction * vh[0, 0]
                y_direction = y_direction * vh[0, 1]
                if x_direction != 0:
                    sign = tf.math.sign(x_direction)
                else:
                    sign = tf.math.sign(y_direction)

                # v_cat.append(vh[0])
                tun_vect = vh[0] * sign
                v_cat.append(tun_vect)

            v_cat = tf.convert_to_tensor(v_cat, dtype=tf.float64)

            tun_vectors.append(v_cat)
        # no point in the category
        else:
            tun_vectors.append(tf.zeros((n_feat_maps, n_dim), dtype=tf.float64))

    return tf.convert_to_tensor(tun_vectors, dtype=tf.float64, name="tun_vectors")


def compute_projections(x, tun_vectors) -> np.array:
    """

    :param x: (n_img, n_feat_map, n_dim)
    :param tun_vectors: (n_cat, n_feat_map, n_dim)
    :param nu:
    :return:
    """
    # case where there's no entry in x
    if x.ndim == 2:
        return tf.zeros((0, x.shape[0], x.shape[1]), dtype=tf.float64)

    # batch per ft_map (meaning putting ft_map dim in first column)
    x = tf.experimental.numpy.moveaxis(x, 0, 1)
    tun_vect = tf.experimental.numpy.moveaxis(tun_vectors, 0, -1)
    projections = tf.matmul(x, tun_vect)  # does not take care of norm_t == 0
    # put back ft_map dim in the middle -> (n_img, n_feat_map, n_dim)
    projections = tf.experimental.numpy.moveaxis(projections, 1, 0)

    return projections


def compute_distances(x: tf.Tensor, radius: tf.Tensor):
    """
    :param x: (n_img, n_feat_map, n_dim)
    :return
    """
    if x.ndim == 2:
        return tf.zeros((0, x.shape[0], x.shape[1]), dtype=tf.float64)
    return tf.exp(- radius * tf.norm(x, axis=2))


def compute_loss_without_ref(proj: tf.Tensor, y: tf.Tensor):
    """

    :param proj: (n_img, n_ft_maps, n_cat)
    :param y: (n_img, )
    :return:
    """

    # compute sum of all exp proj
    # sum_proj = np.sum(np.exp(proj), axis=1)  # don't think this is the correct way for us (email)
    n_img = proj.shape[0]

    # treat neutral as own category
    loss = 0
    for i in range(n_img):
        enumerator = tf.exp(proj[i, :, int(y[i])])
        denominator = tf.reduce_sum(tf.exp(proj[i]), axis=-1)
        loss += tf.reduce_sum(enumerator / denominator)

    return -loss


def compute_loss_with_ref(proj: tf.Tensor, y: tf.Tensor, distances: tf.Tensor):
    """

    :param proj: (n_img, n_ft_maps, n_cat)
    :param y: (n_img, )
    :param distances: (n_img, n_ft_maps)
    :return:
    """

    # remove first column (ref column)
    proj = proj[..., 1:]

    # compute sum of all exp proj
    # sum_proj = np.sum(np.exp(proj), axis=1)  # don't think this is the correct way for us (email)
    n_img = proj.shape[0]

    # treat neutral as own category
    loss = 0
    for i in range(n_img):
        # ref sample
        if int(y[i]) == 0:
            loss += tf.reduce_sum(distances[i])
        # cat sample
        else:
            enumerator = tf.exp(proj[i, :, int(y[i])-1])
            denominator = tf.reduce_sum(tf.exp(proj[i]), axis=-1)
            loss += tf.reduce_sum((1 - distances[i]) * enumerator / denominator)

    return -loss


# @tf.function  # create a graph (non-eager mode!)
def optimize_NRE(x, y, n_cat, batch_size=32, n_ref=1, init_ref=None, lr=0.01, n_epochs=20, do_plot=False):
    """

    :param x: (n_pts, n_feature_maps, n_dim)
    :param y:
    :param neutral:
    :return:
    """
    if x.ndim == 2:
        x = tf.expand_dims(x, axis=1)

    n_dim = np.shape(x)[-1]
    n_feat_maps = np.shape(x)[1]

    # initialize trainable parameters
    shifts = tf.zeros((n_ref, n_feat_maps, n_dim), dtype=tf.float64, name="shifts")
    if init_ref is not None:
        shifts = tf.identity(init_ref, name="shifts")
    radius = tf.ones((n_ref, n_feat_maps), dtype=tf.float64, name="radius")
    print("shape shifts", shifts.shape)
    print("shape radius", radius.shape)

    # declare sequence parameters
    if do_plot:
        path = ""
        video_name = "NRE_loss_training.mp4"
        n_rows = int(np.sqrt(n_feat_maps))
        n_columns = np.ceil(n_feat_maps / n_rows).astype(int)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(path, video_name), fourcc, 30, (n_columns * 400, n_rows * 400))

    for epoch in range(n_epochs):
        it = 0
        for x_batch, y_batch in batch(x, y, n=batch_size):
            #print("shape x_batch", x_batch.shape, "shape y_batch", y_batch.shape)
            loss = 0
            with tf.GradientTape() as tape:
                tape.watch(shifts)
                tape.watch(radius)
                for r in range(n_ref):
                    # filter data per ref
                    ref_mask = tf.equal(y_batch[:, 1], r)
                    x_filt = tf.gather(x_batch, tf.squeeze(tf.where(ref_mask)))
                    y_filt = tf.gather(y_batch[:, 0], tf.squeeze(tf.where(ref_mask)))

                    # subtract  shifts to x
                    x_shifted = tf.subtract(x_filt, shifts[r], name="x_shifted")
                    # print("shape x_shifted", x_shifted.shape)

                    # get tun vectors
                    tun_vectors = compute_tun_vectors(x_shifted, y_filt, n_cat)
                    # print("tun_vectors", tun_vectors.shape)
                    # print(tun_vectors)

                    # # get projections
                    projections = compute_projections(x_shifted, tun_vectors)
                    # print("projections", projections.shape)
                    # print(projections)

                    if n_ref > 1:
                        distances = compute_distances(x_shifted, radius[r])
                        # print("distances", distances.shape)

                        # compute loss
                        loss += compute_loss_with_ref(projections, y_filt, distances)
                    else:
                        # compute loss
                        loss += compute_loss_without_ref(projections, y_filt)
            # print(f"{epoch} loss {loss}, radius[0]: {radius[0]}", end='\r')
            print(f"{epoch}, it: {it}, loss {loss}", end='\r')

            # compute gradients
            grad_shifts, grad_radius = tape.gradient(loss, [shifts, radius])
            # print("grad shifts", grad_shifts.shape)

            # update parameters
            shifts = shifts - lr * grad_shifts
            radius = radius - lr * grad_radius
            # print(f"{epoch} shifts {shifts}")
            # print()

            if do_plot:
                tun_vect = tun_vectors.numpy()
                # img = plot_space(x.numpy(), y.numpy(), n_cat, shifts=shifts.numpy(), tun_vectors=tun_vect)
                img = plot_space(x_batch.numpy(), y_batch.numpy(), n_cat,
                                 shifts=shifts.numpy(),
                                 tun_vectors=tun_vect)

                # write image
                video.write(img)

            # increase iteration
            it += 1

    if do_plot:
        cv2.destroyAllWindows()
        video.release()

    # print last one to keep in the log
    print(f"{epoch} it: {it}, loss {loss}, radius: {radius}")


if __name__ == '__main__':
    profiler = False
    do_plot = True

    # declare parameters
    n_dim = 2
    n_cat = 4
    neutral_cat = None
    n_points = 5
    n_latent = 3
    n_ref = 2
    n_entry = n_points * n_cat * n_ref
    print(f"{n_entry} entry created!")

    # define configuration
    config_file = 'NR_03_FERG_from_LMK_m0001.json'
    # load config
    config = load_config(config_file, path='/Users/michaelstettler/PycharmProjects/BVS/BVS/configs/norm_reference')
    print("-- Config loaded --")
    print()

    # Load data
    train_data = load_data(config, get_raw=True)
    train_label = train_data[1]
    test_data = load_data(config, train=False, get_raw=True)
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
    print("shape y_train", np.shape(y_train))
    print("shape y_test", np.shape(y_test))

    # transform to tensor
    init_ref = None
    # init_ref = tf.convert_to_tensor(x_train[[0, 20]] + 0.01, dtype=tf.float64)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float64)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
    print("shape x_train", x_train.shape)
    print("shape y_train", y_train.shape)

    if profiler:
        # create logs and tensorboard
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs/func/%s' % stamp
        writer = tf.summary.create_file_writer(logdir)

        tf.summary.trace_on(graph=True, profiler=True)

    # optimize_NRE
    optimize_NRE(x_train, y_train, n_cat,
                 batch_size=512,
                 n_ref=n_ref,
                 init_ref=init_ref,
                 lr=0.1,
                 n_epochs=200,
                 do_plot=do_plot)

    if profiler:
        with writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir)
