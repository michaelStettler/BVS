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


def generate_data_set(n_dim: int, n_cat: int, n_points: int, min_length=2, max_length=5, ref_at_origin=True,
                      n_latent=1, n_ref=1, variance_ratio=1, ref_variance=1, balanced=True, do_plot=False):
    """

    :param n_dim:
    :param n_cat:
    :param n_points: n_points per category
    :param ref_at_0:
    :param balanced:
    :return:
    """

    positions = []

    # for each n_latent, construct n_ref distributions
    for i in range(n_latent):
        ref_positions = []

        # create randomly random direction (phi angles)
        # this is fixed per latent space
        phis = np.random.rand(n_cat - 1) * 2 * np.pi

        for r in range(n_ref):
            # set the ref
            if ref_at_origin and n_ref == 1:
                ref_origin = np.zeros(n_dim)
            else:
                ref_origin = (np.random.rand(n_dim) - 0.5) * ref_variance

            # create random positions around the center (ref_origin)
            ref_pos = np.random.rand(n_points, n_dim) * variance_ratio + ref_origin - 0.5

            # create randomly different length between min and max length
            lengths = np.random.rand(n_cat - 1) * (max_length - min_length) + min_length

            # compute xy coordinates for each direction
            tun_refs = np.array([np.cos(phis), np.sin(phis)]).T * np.expand_dims(lengths, axis=1)

            # generate clouds of positions for each category (origin centered)
            tun_pos = np.random.rand(n_cat - 1, n_points, n_dim) - 0.5

            # translate to tuning positions
            tun_pos += np.repeat(np.expand_dims(tun_refs, axis=1), n_points, axis=1) + ref_origin

            # create pos
            position = np.concatenate((ref_pos, np.reshape(tun_pos, (-1, n_dim))), axis=0)

            # append to ref
            ref_positions.append(position)

        # remove extra dim if only one ref
        ref_positions = np.squeeze(ref_positions)
        if ref_positions.ndim == 3:
            ref_positions = np.reshape(ref_positions, (-1, ref_positions.shape[2]))

        # construct dataset
        positions.append(ref_positions)

    # return array as either (n_pts, n_dim) if n_latent ==1, or else as (n_pts, n_latent, n_dim)
    positions = np.array(positions)
    positions = np.moveaxis(positions, 0, 1)
    positions = np.squeeze(positions)

    # construct label
    labels = []
    for r in range(n_ref):
        for i in range(len(ref_pos)):
            labels.append([0, r])
        for i in range(n_cat - 1):
            for j in range(len(tun_pos[i])):
                labels.append([i + 1, r])

    return positions, np.array(labels).astype(int)


def plot_space(positions, labels, n_cat, ref_vector=[0, 0], tun_vectors=None, min_length=5, max_length=5,
               shifts=None, show=False):

    # retrieve variables
    n_pts = positions.shape[0]
    n_feat_map = positions.shape[1]
    n_ref = 1
    if shifts is not None:
        n_ref = shifts.shape[0]

    # set img size
    n_rows = int(np.sqrt(n_feat_map))
    n_columns = np.ceil(n_feat_map / n_rows).astype(int)
    image = np.zeros((n_rows*400, n_columns*400, 3))

    # plot
    cmap = cm.get_cmap('viridis', n_cat)
    for ft in range(n_feat_map):
        fig = plt.figure(figsize=(4, 4), dpi=100)  # each subplot is 400x400

        for r in range(n_ref):
            # shifts positions
            ref_idx = np.arange(n_pts)  # construct all index for commmon pipeline
            if shifts is not None:
                ref_idx = ref_idx[labels[:, 1] == r]
                pos_ft = positions[ref_idx, ft] - shifts[r, ft]
            else:
                pos_ft = positions[:, ft]

            # get unique labels
            uniques = np.unique(labels)

            for i, color in zip(range(n_cat), cmap(range(n_cat))):
                label = uniques[i]

                label_name = None
                if r == 0:
                    label_name = "cat {}".format(label)
                    if label == 0:
                        label_name = "reference"

                # get all positions for this label
                pos = pos_ft[labels[ref_idx, 0] == label]

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


def compute_tun_vectors(x, y, n_cat):
    n_feat_maps = np.shape(x)[1]

    tun_vectors = []
    # for each expression
    for cat in range(n_cat):
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


def compute_distances(x: tf.Tensor, radius: tf.Tensor):
    """
    :param x: (n_img, n_feat_map, n_dim)
    :return
    """
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
        for x_batch, y_batch in batch(x, y, n=batch_size):
            print("shape x_batch", x_batch.shape, "shape y_batch", y_batch.shape)
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
                    # print("tun_vectors", tun_vectors.shape)  # working (2, 1, 1) new (2, 1, 2)
                    # print(tun_vectors)

                    # # get projections
                    projections = compute_projections(x_shifted, tun_vectors)
                    # print("projections", projections.shape)  # working (10, 2) new (2, 1, 2)
                    # print(projections)

                    if n_ref > 1:
                        distances = compute_distances(x_shifted, radius[r])
                        # print("distances", distances.shape)

                        # compute loss
                        loss += compute_loss_with_ref(projections, y_filt, distances)
                    else:
                        # compute loss
                        loss += compute_loss_without_ref(projections, y_filt)
            print(f"{epoch} loss {loss}, radius[0]: {radius[0]}", end='\r')

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
                img = plot_space(x.numpy(), y.numpy(), n_cat, shifts=shifts.numpy(), tun_vectors=tun_vect)

                # write image
                video.write(img)

        # print last one to keep in the log
        print(f"{epoch} loss {loss}, radius: {radius}")

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
    y_train = [train_label, train_avatar]
    print("shape y_train", np.shape(y_train))

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
                 batch_size=64,
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
