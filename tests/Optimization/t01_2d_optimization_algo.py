import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import cm
from datetime import datetime

viridis = cm.get_cmap('viridis', 12)

np.random.seed(1)

"""
run: python -m tests.Optimization.t01_2d_optimization_algo
tensorboard: tensorboard --logdir logs/func
"""


def generate_data_set(n_dim: int, n_cat: int, n_points: int, min_length=2, max_length=5, ref_at_origin=True,
                      balanced=True, do_plot=False):
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
    positions = np.concatenate((ref_pos, np.reshape(tun_pos, (-1, n_dim))), axis=0)
    # construct label
    labels = []
    for i in range(len(ref_pos)):
        labels.append(0)
    for i in range(n_cat - 1):
        for j in range(len(tun_pos[i])):
            labels.append(i + 1)

    return positions, np.array(labels).astype(int)


def plot_space(positions, labels, n_cat, ref_vector=[0, 0], tun_vectors=None, min_length=5, max_length=5):
    uniques = np.unique(labels)

    # plot only first feature map
    pos_ft = positions[:, 0]

    # plot
    cmap = cm.get_cmap('viridis', n_cat)
    plt.figure()
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

        if x_direction * tun_vectors[i - 1, 0] < 0:
            sign = -1
        else:
            sign = 1

        # plot the positions
        plt.scatter(pos[:, 0], pos[:, 1], color=color, label=label_name)

        if i != 0 and tun_vectors is not None and ref_vector is not None:
            plt.plot([ref_vector[0], sign * max_length * tun_vectors[i - 1, 0]], [ref_vector[1], sign * max_length * tun_vectors[i - 1, 1]], color=color)

    if ref_vector is not None:
        # plot cross
        plt.plot(ref_vector[0], ref_vector[1], 'xk')


    plt.legend()
    plt.axis([-min_length, max_length, -min_length, max_length])
    plt.show()


def compute_tun_vectors(x, y, n_cat):
    n_dim = np.shape(x)[-1]
    n_feat_maps = np.shape(x)[1]
    print("n_dim: {}, n_cat: {}".format(n_dim, n_cat))

    tun_vectors = []
    # for each expression
    for cat in range(1, n_cat):
        print("cat", cat)
        # construct mask to get slice of tensor x per category (in numpy x[y == cat])
        bool_mask = tf.equal(y, cat)
        bool_mask = tf.repeat(tf.expand_dims(bool_mask, axis=1), x.shape[1], axis=1)
        bool_mask = tf.repeat(tf.expand_dims(bool_mask, axis=2), x.shape[2], axis=2)
        x_cat = tf.gather(x, tf.where(bool_mask))
        print("shape x_cat", x_cat.shape)

        # declare tuning vect per category
        # v_cat = tf.zeros((n_feat_maps, n_dim))
        v_cat = []
        for f in range(n_feat_maps):
            u, s, vh = tf.linalg.svd(x_cat[:, f], full_matrices=True)
            print("shape u, s, vh", np.shape(u), np.shape(s), np.shape(vh))
            print("shape vh[0]", np.shape(vh[0]))
            # v_cat[f] = vh[0] + ref_vectors[f]
            # v_cat.append(vh[0] + ref_vectors[f])
            v_cat.append(vh[0, 0])

        v_cat = tf.convert_to_tensor(v_cat, dtype=tf.float64)
        print("shape v_cat", v_cat.shape)

        tun_vectors.append(v_cat)

    return tf.convert_to_tensor(tun_vectors, dtype=tf.float64, name="tun_vectors")


def compute_projections(x, tun_vectors, nu=1, neutral_threshold=0) -> np.array:
    """

    :param x: (n_img, n_feat_map, n_dim)
    :param ref_vectors: (n_feat_map, n_dim)
    :param tun_vectors: (n_cat, n_feat_map, n_dim)
    :param nu:
    :param neutral_threshold:
    :param verbose:
    :return:
    """

    # normalize by norm of each landmark
    # norm_t = np.linalg.norm(tun_vectors, axis=2)  # (n_cat, n_feat_map)

    # vectorized version
    # diff = x - ref_vectors
    # projections = np.dot(diff, np.moveaxis(tun_vectors, 0, -1)) / norm_t  # does not take care of norm_t == 0
    # projections = np.dot(x, np.moveaxis(tun_vectors, 0, -1)) / norm_t  # does not take care of norm_t == 0
    # projections = np.power(projections, nu)
    # projections[projections < 0] = 0
    # projections = projections[..., 0, :]  # remove duplicate from the 3 by 3,
    # todo: is there not a better way to do the dot product?
    # projections = np.sum(projections, axis=1)  # sum of feature maps

    # ALEX's version
    tun_vect = tf.experimental.numpy.moveaxis(tun_vectors, 0, -1)
    projections = tf.matmul(x, tun_vect)  # does not take care of norm_t == 0
    projections = tf.math.pow(projections, nu)
    projections = tf.reduce_sum(projections, axis=1, name="projections")  # sum of feature maps

    # # apply neutral threshold
    # projections[projections < neutral_threshold] = 0

    return projections


def compute_loss(proj: tf.Tensor, y: tf.Tensor, n_cat: int) -> float:
    """

    :param proj: (n_img, n_cat)
    :param y: (n_img, )
    :return:
    """

    # compute sum of all exp proj
    # sum_proj = np.sum(np.exp(proj), axis=1)  # don't think this is the correct way for us (email)

    loss = 0
    for i in range(1, n_cat):
        loss += tf.exp(proj[i, int(y[i]) - 1])

    return -loss


# @tf.function  # create a graph (non-eager mode!)
def optimize_NRE(x, y, n_cat, neutral=0, lr=0.1, n_epochs=1):
    """

    :param x: (n_pts, n_feature_maps, n_dim)
    :param y:
    :param neutral:
    :return:
    """
    n_dim = np.shape(x)[-1]
    n_feat_maps = np.shape(x)[1]

    # initialize trainable parameters
    shifts = tf.zeros((n_feat_maps, n_dim), dtype=tf.float64, name="shifts")
    radius = tf.ones(n_feat_maps, dtype=tf.float64, name="radius") * 0.01
    print("shape shifts", shifts.shape)
    print("shape radius", radius.shape)

    for epoch in range(n_epochs):
        with tf.GradientTape() as tape:
            tape.watch(shifts)
            tape.watch(radius)

            # get tun vectors
            x_shifted = tf.subtract(x, shifts, name="x_shifted")
            tun_vectors = compute_tun_vectors(x_shifted, y, n_cat)
            print("tun_vectors", tun_vectors.shape)
            # print(tun_vectors)

            # # get projections
            projections = compute_projections(x_shifted, tun_vectors)
            print("projections", projections.shape)
            # print(projections)

            # compute loss
            loss = compute_loss(projections, y, n_cat)
            print("loss", loss)

        # update parameters
        grad_shifts = tape.gradient(loss, x_shifted)
        print("grad shifts", grad_shifts.shape)
        shifts = shifts - lr * grad_shifts
        #
        # grad_rad = tape.gradient(loss, radius)
        # radius = radius - lr * grad_rad
        #
        tun_vect = tun_vectors.numpy()
        plot_space(x.numpy(), y.numpy(), n_cat, tun_vectors=tun_vect[:, 0])


if __name__ == '__main__':
    # declare parameters
    n_dim = 2
    n_cat = 4  # (3 + neutral)
    n_points = 5
    # generate random data
    x_train, y_train = generate_data_set(n_dim, n_cat, n_points, ref_at_origin=False)
    print("shape x_train", np.shape(x_train))
    print("shape y_train", np.shape(y_train))

    # # plot generated data
    # plot_space(x_train, y_train)
    # transform to tensor
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float64)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
    print("shape x_train", x_train.shape)
    print("shape y_train", y_train.shape)


    # create logs and tensorboard
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/func/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)

    # optimize_NRE
    tf.summary.trace_on(graph=True, profiler=True)
    optimize_NRE(tf.expand_dims(x_train, axis=1), y_train, n_cat)

    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)
