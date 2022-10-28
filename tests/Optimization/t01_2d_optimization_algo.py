import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

viridis = cm.get_cmap('viridis', 12)

np.random.seed(1)

"""
run: python -m tests.Optimization.t01_2d_optimization_algo
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


def plot_space(positions, labels, ref_vector=None, tun_vectors=None, min_length=5, max_length=5):
    uniques = np.unique(labels)
    n_cat = len(uniques)

    # plot
    cmap = cm.get_cmap('viridis', n_cat)
    plt.figure()
    for i, color in zip(range(n_cat), cmap(range(n_cat))):
        label = uniques[i]

        label_name = "cat {}".format(label)
        if label == 0:
            label_name = "reference"

        # get all positions for this label
        pos = positions[labels == label]

        # scale tuning vector for plotting
        max_length = np.max(np.linalg.norm(pos, axis=1))
        x_direction = pos[np.argmax(np.linalg.norm(pos, axis=1))][0]
        if x_direction * tun_vectors[i - 1, 0] < 0:
            sign = -1
        else:
            sign = 1

        # plot the positions
        plt.scatter(pos[:, 0], pos[:, 1], color=color, label=label_name)

        if i != 0 and tun_vectors is not None and ref_vector is not None:
            plt.plot([ref_vector[0], sign * max_length * tun_vectors[i - 1, 0]], [ref_vector[1], sign * max_length * tun_vectors[i - 1, 1]], color=color)

    if ref_vector is not None:
        print("ref_vector", ref_vector)
        # plot cross
        plt.plot(ref_vector[0], ref_vector[1], 'xk')


    plt.legend()
    plt.axis([-min_length, max_length, -min_length, max_length])
    plt.show()


def compute_projections(x, ref_vectors, tun_vectors, nu=1, neutral_threshold=0) -> np.array:
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
    norm_t = np.linalg.norm(tun_vectors, axis=2)  # (n_cat, n_feat_map)

    # vectorized version
    diff = x - ref_vectors
    projections = np.dot(diff, np.moveaxis(tun_vectors, 0, -1)) / norm_t  # does not take care of norm_t == 0
    projections = np.power(projections, nu)
    projections[projections < 0] = 0
    projections = projections[..., 0, :]  # remove duplicate from the 3 by 3,
    # todo: is there not a better way to do the dot product?
    projections = np.sum(projections, axis=1)  # sum of feature maps

    # apply neutral threshold
    projections[projections < neutral_threshold] = 0

    return projections


def compute_loss(proj: np.array, y: np.array) -> float:
    """

    :param proj: (n_img, n_cat)
    :param y: (n_img, )
    :return:
    """

    # compute sum of all exp proj
    sum_proj = np.sum(np.exp(proj), axis=1)

    loss = 0
    for i in range(len(proj)):
        if y[i] != 0:  # assume neutral == 0
            loss += np.exp(proj[i, y[i] - 1]) / sum_proj[i]

    return loss


def optimize_NRE(x: np.array, y, neutral=0, lr=0.1):
    """

    :param x: (n_pts, n_feature_maps, n_dim)
    :param y:
    :param neutral:
    :return:
    """
    n_dim = np.shape(x)[-1]
    uniques = np.unique(y)
    n_cat = len(uniques)
    n_feat_maps = np.shape(x)[1]
    print("n_dim: {}, n_cat: {}".format(n_dim, n_cat))

    ref_vectors = np.zeros((n_feat_maps, n_dim))
    print("shape ref_vectors", np.shape(ref_vectors))
    tun_vectors = np.zeros((n_cat - 1, n_feat_maps, n_dim))
    print("shape tun_vectors", np.shape(tun_vectors))

    # for each expression
    c = 0
    for cat in uniques:
        if cat != neutral:
            x_cat = x[y == cat]
            v_cat = np.zeros((n_feat_maps, n_dim))
            for f in range(n_feat_maps):
                u, s, vh = np.linalg.svd(x_cat[:, f])
                print("shape u, s, vh", np.shape(u), np.shape(s), np.shape(vh))
                print("s", s)
                print("shape vh[0]", np.shape(vh[0]))
                v_cat[f] = vh[0] + ref_vectors[f]

            tun_vectors[c] = v_cat
            c += 1

    print("tun_vectors", np.shape(tun_vectors))
    print(tun_vectors)

    # get projections
    projections = compute_projections(x, ref_vectors, tun_vectors)
    print("projections", np.shape(projections))

    # compute loss
    loss = compute_loss(projections, y)
    print("loss", loss)

    # move ref vector (do something like)
    ref_vectors += lr * loss

    plot_space(x_train, y_train, ref_vector=ref_vectors[0], tun_vectors=tun_vectors[:, 0])


if __name__ == '__main__':
    # declare parameters
    n_dim = 2
    n_cat = 4  # (3 + random)
    n_points = 5
    # generate random data
    x_train, y_train = generate_data_set(n_dim, n_cat, n_points, ref_at_origin=False)
    print("shape x_train", np.shape(x_train))
    print("shape y_train", np.shape(y_train))

    # # plot generated data
    # plot_space(x_train, y_train)

    # optimize_NRE
    optimize_NRE(np.expand_dims(x_train, axis=1), y_train)
