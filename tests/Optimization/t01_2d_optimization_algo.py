import os
import numpy as np

import cv2
import tensorflow as tf
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

viridis = cm.get_cmap('viridis', 12)
matplotlib.use('agg')

np.random.seed(1)

"""
run: python -m tests.Optimization.t01_2d_optimization_algo
tensorboard: tensorboard --logdir logs/func
"""


def generate_data_set(n_dim: int, n_cat: int, n_points: int, min_length=2, max_length=5, ref_at_origin=True,
                      n_latent=1, balanced=True, do_plot=False):
    """

    :param n_dim:
    :param n_cat:
    :param n_points: n_points per category
    :param ref_at_0:
    :param balanced:
    :return:
    """
    if ref_at_origin:
        ref_origin = np.zeros(n_dim)
    else:
        ref_origin = np.random.rand(n_dim)

    positions = []

    for i in range(n_latent):
        # create random positions around the center (ref_origin)
        ref_pos = np.random.rand(n_points, n_dim) + ref_origin - 0.5

        # create randomly random direction (phi angles)
        phis = np.random.rand(n_cat - 1) * 2 * np.pi

        # create randomly different length between min and max length
        lengths = np.random.rand(n_cat - 1) * (max_length - min_length) + min_length

        # compute xy coordinates for each direction
        tun_refs = np.array([np.cos(phis), np.sin(phis)]).T * np.expand_dims(lengths, axis=1)

        # generate clouds of positions for each category (origin centered)
        tun_pos = np.random.rand(n_cat - 1, n_points, n_dim) - 0.5

        # translate to tuning positions
        tun_pos += np.repeat(np.expand_dims(tun_refs, axis=1), n_points, axis=1) + ref_origin

        # construct dataset
        positions.append(np.concatenate((ref_pos, np.reshape(tun_pos, (-1, n_dim))), axis=0))

    # return array as either (n_pts, n_dim) if n_latent ==1, or else as (n_pts, n_latent, n_dim)
    positions = np.array(positions)
    positions = np.moveaxis(positions, 0, 1)
    positions = np.squeeze(positions)

    # construct label
    labels = []
    for i in range(len(ref_pos)):
        labels.append(0)
    for i in range(n_cat - 1):
        for j in range(len(tun_pos[i])):
            labels.append(i + 1)

    return positions, np.array(labels).astype(int)


def plot_space(positions, labels, n_cat, ref_vector=[0, 0], tun_vectors=None, min_length=5, max_length=5,
               show=False):
    uniques = np.unique(labels)

    # retrieve variables
    n_feat_map = positions.shape[1]

    # set img size
    n_rows = int(np.sqrt(n_feat_map))
    n_columns = np.ceil(n_feat_map / n_rows).astype(int)
    image = np.zeros((n_rows*400, n_columns*400, 3))

    # plot
    cmap = cm.get_cmap('viridis', n_cat)
    for ft in range(n_feat_map):
        fig = plt.figure(figsize=(4, 4), dpi=100)  # each subplot is 400x400
        pos_ft = positions[:, ft]

        for i, color in zip(range(n_cat), cmap(range(n_cat))):
            label = uniques[i]

            label_name = "cat {}".format(label)
            if label == 0:
                label_name = "reference"

            # get all positions for this label
            pos = pos_ft[labels == label]

            # scale tuning vector for plotting
            pos_norm = np.linalg.norm(pos, axis=1)
            max_length = np.max(pos_norm)
            x_direction = pos[np.argmax(pos_norm)][0]

            sign = np.sign(x_direction * tun_vectors[i, ft, 0])

            # plot the positions
            plt.scatter(pos[:, 0], pos[:, 1], color=color, label=label_name)

            # plot tuning line
            if tun_vectors is not None and ref_vector is not None:
                # x_ = [ref_vector[ft, 0], sign * max_length * tun_vectors[i, 0]]
                x_ = [0, tun_vectors[i, ft, 0]]
                # y_ = [ref_vector[ft, 1], sign * max_length * tun_vectors[i, 1]]
                y_ = [0, tun_vectors[i, ft, 1]]
                plt.plot(x_, y_, color=color)

        # plot cross
        plt.plot(0, 0, 'xk')
        # plt.plot(ref_vector[ft, 1], ref_vector[ft, 0], 'xk')

        if ft == 0:
            plt.legend()
        # plt.axis([-min_length, max_length, -min_length, max_length])
        plt.axis([-10, 10, -10, 10])

        if show:
            plt.show()

        # transform figure to numpy array
        fig.canvas.draw()
        fig.tight_layout(pad=0)

        # compute row/col number
        n_col = ft % n_columns
        n_row = ft % n_rows

        # transform figure to numpy and append it in the correct place
        figure = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image[n_row*400:(n_row+1)*400, n_col*400:(n_col+1)*400] = figure

        # clear figure
        plt.cla()
        plt.clf()
        plt.close()

    return image.astype(np.uint8)


def compute_tun_vectors(x, y, n_cat, neutral_cat=None):
    n_dim = np.shape(x)[-1]
    n_feat_maps = np.shape(x)[1]
    start = 0

    tun_vectors = []
    # add neutral direction if neutral
    if neutral_cat:
        start = 1
        neutral_tuning = np.zeros((n_feat_maps, n_dim))
        tun_vectors.append(neutral_tuning)

    # for each expression
    for cat in range(start, n_cat):
        # construct mask to get slice of tensor x per category (in numpy x[y == cat])
        bool_mask = tf.equal(y, cat)
        x_cat = tf.gather(x, tf.squeeze(tf.where(bool_mask)))

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

    return tf.convert_to_tensor(tun_vectors, dtype=tf.float64, name="tun_vectors")


def compute_projections(x, tun_vectors) -> np.array:
    """

    :param x: (n_img, n_feat_map, n_dim)
    :param tun_vectors: (n_cat, n_feat_map, n_dim)
    :param nu:
    :return:
    """

    # batch per ft_map (meaning putting ft_map dim in first column)
    x = tf.experimental.numpy.moveaxis(x, 0, 1)
    tun_vect = tf.experimental.numpy.moveaxis(tun_vectors, 0, -1)
    projections = tf.matmul(x, tun_vect)  # does not take care of norm_t == 0
    # put back ft_map dim in the middle -> (n_img, n_feat_map, n_dim)
    projections = tf.experimental.numpy.moveaxis(projections, 1, 0)

    return projections


def compute_loss(proj: tf.Tensor, y: tf.Tensor, neutral_cat=None, eps=1e-3) -> float:
    """

    :param proj: (n_img, n_cat)
    :param y: (n_img, )
    :return:
    """

    # compute sum of all exp proj
    # sum_proj = np.sum(np.exp(proj), axis=1)  # don't think this is the correct way for us (email)

    if neutral_cat:
        raise NotImplementedError

    # treat neutral as own category
    loss = 0
    for i in range(proj.shape[0]):
        enumerator = tf.exp(proj[i, :, int(y[i])])
        denominator = tf.reduce_sum(tf.exp(proj[i]), axis=-1)
        loss += tf.reduce_sum(enumerator / denominator)

    return -loss


# @tf.function  # create a graph (non-eager mode!)
def optimize_NRE(x, y, n_cat, neutral_cat=None, lr=0.01, n_epochs=20, do_plot=False):
    """

    :param x: (n_pts, n_feature_maps, n_dim)
    :param y:
    :param neutral:
    :return:
    """
    n_dim = np.shape(x)[-1]
    n_feat_maps = np.shape(x)[1]

    # initialize trainable parameters
    # shifts = tf.ones((n_feat_maps, n_dim), dtype=tf.float64, name="shifts")
    shifts = tf.zeros((n_feat_maps, n_dim), dtype=tf.float64, name="shifts")
    radius = tf.ones(n_feat_maps, dtype=tf.float64, name="radius") * 0.01
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
        with tf.GradientTape() as tape:
            tape.watch(shifts)
            tape.watch(radius)

            x_shifted = tf.subtract(x, shifts, name="x_shifted")
            # print("shape x_shifted", x_shifted.shape)

            # get tun vectors
            tun_vectors = compute_tun_vectors(x_shifted, y, n_cat, neutral_cat=neutral_cat)
            # print("tun_vectors", tun_vectors.shape)  # working (2, 1, 1) new (2, 1, 2)
            # print(tun_vectors)

            # # get projections
            projections = compute_projections(x_shifted, tun_vectors)
            # print("projections", projections.shape)  # working (10, 2) new (2, 1, 2)
            # print(projections)

            # compute loss
            loss = compute_loss(projections, y, neutral_cat=neutral_cat)
            print(f"{epoch} loss {loss}", end='\r')

        # update parameters
        grad_shifts = tape.gradient(loss, shifts)
        # print("grad shifts", grad_shifts.shape)
        shifts = shifts - lr * grad_shifts
        # print(f"{epoch} shifts {shifts}")
        # print()

        if do_plot:
            tun_vect = tun_vectors.numpy()
            img = plot_space(x_shifted.numpy(), y.numpy(), n_cat, tun_vectors=tun_vect)

            # write image
            video.write(img)

    if do_plot:
        cv2.destroyAllWindows()
        video.release()


if __name__ == '__main__':
    profiler = False
    do_plot = True

    # declare parameters
    n_dim = 2
    n_cat = 4
    neutral_cat = None
    n_points = 5
    n_latent = 3
    # generate random data
    x_train, y_train = generate_data_set(n_dim, n_cat, n_points, ref_at_origin=False, n_latent=n_latent)
    print("shape x_train", np.shape(x_train))
    print("shape y_train", np.shape(y_train))
    # x_train = [[0, 2], [3, 1], [3, 2], [-1, -1]]
    # y_train = [0, 1, 1, 2]
    # print("shape x_train", np.shape(x_train))


    # plot generated data
    # plot_space(x_train, y_train)

    # transform to tensor
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
    if x_train.ndim == 2:
        x_train = tf.expand_dims(x_train, axis=1)

    optimize_NRE(x_train, y_train, n_cat,
                 lr=0.1,
                 n_epochs=50,
                 neutral_cat=neutral_cat,
                 do_plot=do_plot)

    if profiler:
        with writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir)
