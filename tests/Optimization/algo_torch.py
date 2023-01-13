import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

viridis = cm.get_cmap('viridis', 12)

np.random.seed(1)

"""
run: python -m tests.Optimization.algo_torch
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

def center_reference(x: np.array, ref_vectors: np.array) -> np.array:
    """
    Centers the data s.t. the reference is the origin. Necessary to work in a linear subspace.
    Only needs to be called ONCE in the beginning, not on every iteration
    :param x: (n_img, n_feat_map, n_dim)
    :param ref_vectors: (n_feat_map, n_dim)
    :return: (n_img, n_feat_map, n_dim)
    """
    return x - ref_vectors


def plot_space(positions, labels, ref_vector=None, tun_vectors=None, radius=None, min_length=5, max_length=5, save=False, name=None):
    uniques = np.unique(labels)
    n_cat = len(uniques)

    positions, labels = positions.clone().detach().data.cpu().numpy(), labels.clone().detach().data.cpu().numpy()
    tun_vectors = tun_vectors.clone().detach().data.cpu().numpy()
    radius = radius.clone().detach().data.cpu().numpy()
    positions = np.squeeze(positions)
    # plot
    cmap = cm.get_cmap('viridis', n_cat)
    plt.figure()
    r = radius
    circle_x = np.linspace(0, r, 1000)
    circle_y = (np.sqrt(r**2 -circle_x**2))
    plt.scatter(circle_x, circle_y, s=1, color="black")
    plt.scatter(circle_x, -circle_y, s=1, color="black")
    plt.scatter(-circle_x, circle_y, s=1, color="black")
    plt.scatter(-circle_x, -circle_y, s=1, color="black")
    for i, color in zip(range(n_cat), cmap(range(n_cat))):
        label = uniques[i]

        label_name = "cat {}".format(label)
        if label == 0:
            label_name = "reference"

        # get all positions for this label
        pos = positions[labels == label]

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

        ref_vector = [0, 0]
        if i != 0 and tun_vectors is not None and ref_vector is not None:
            plt.plot([ref_vector[0], sign * max_length * tun_vectors[i - 1, 0]], [ref_vector[1], sign * max_length * tun_vectors[i - 1, 1]], color=color)

    if ref_vector is not None:
        # plot cross
        plt.plot(ref_vector[0], ref_vector[1], 'xk')

    plt.legend()
    plt.axis([-min_length, max_length, -min_length, max_length])
    if save:
        plt.savefig(name)
    plt.show()

def compute_tun_vectors(x, y, neutral):
    print(x.shape)
    n_dim = x.shape[-1]
    uniques = torch.unique(y)
    n_cat = len(uniques)
    n_feat_maps = x.shape[1]
    print("n_dim: {}, n_cat: {}".format(n_dim, n_cat))

    ref_vectors = torch.zeros((n_feat_maps, n_dim))
    print("shape ref_vectors", ref_vectors.shape)
    tun_vectors = torch.zeros((n_cat - 1, n_feat_maps, n_dim), dtype=torch.float64)
    print("shape tun_vectors", tun_vectors.shape)
    # for each expression
    c = 0
    for cat in uniques:
        if cat != neutral:
            x_cat = x[y == cat]
            print("shape x_cat", x_cat.shape)
            v_cat = torch.zeros((n_feat_maps, n_dim))
            for f in range(n_feat_maps):
                u, s, vh = torch.linalg.svd(x_cat[:, f, :])
                print("shape v_cat[f, :] vh", v_cat[f, :].shape, vh.shape)

                # Orient tuning vectors properly
                x_np = x_cat.detach().numpy()
                v_np = vh[0, :].detach().numpy()
                x_direction = x_np[np.argmax(np.linalg.norm(x_np[:, f, :], axis=-1), axis=0), 0][0]
                if x_direction * v_np[0] < 0:
                    sign = -1
                else:
                    sign = 1

                v_cat[f, :] = vh[0, :] * sign
            tun_vectors[c] = v_cat

            c += 1

    return tun_vectors

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

    projections = torch.matmul(x, torch.moveaxis(tun_vectors, 0, -1))
    projections = projections ** nu
    print("projections", projections.shape)

    # todo: Revisit cutting off negative projections. Buggy atm since the tuning vector may have arbitrary sign
    # projections[projections < 0] = 0
    # todo: is there not a better way to do the dot product?
    projections = torch.sum(projections, axis=1)  # sum of feature maps

    ### This should probably be done in the loss function because we will need a differentiable mechanism
    # to determine whether the expression is neutral I think
    # # apply neutral threshold
    # projections[projections < neutral_threshold] = 0

    return projections

def compute_neutral_probability(proj: torch.tensor, radius):
    length = torch.sum(proj, axis=1)
    p_expression = 1 / (1 + torch.exp(radius - length))
    return 1 - p_expression


def compute_loss(proj: np.array, x, y: np.array, radius: float) -> float:
    """
    :param proj: (n_img, n_cat)
    :param y: (n_img, )
    :return:
    """

    losses = torch.zeros(len(proj), dtype=torch.float64)
    for i in range(len(proj)):
        if y[i] == 0:
            length = torch.linalg.norm(x[i, :, :], axis=-1)
            prob_neutral = 1 - 1 / (1 + torch.exp(radius - length))
            losses[i] = prob_neutral * 0
        if y[i] != 0:  # assume neutral == 0
            prob_expression = torch.exp(proj[i, y[i] - 1]) / torch.sum(torch.exp(proj[i, :]))
            # print(proj[i, :])
            # print(prob_expression)
            # length = torch.linalg.norm(x[i, :, :], axis=-1)
            # prob_neutral = 1 - 1 / (1 + torch.exp(radius - length))
            prob_neutral = 0
            losses[i] = prob_expression * (1 - prob_neutral)
            # print(prob_expression)
    print(losses)
    loss = torch.sum(losses)
    print("loss", loss)
    return - loss

# def compute_neutral_loss(x, y, radius):
#     x_neutral = x[y == 0]
#     x_neutral = torch.linalg.norm(x_neutral, axis=-1)
#     x_neutral = 1 / (1 + torch.exp(radius - x_neutral))
#     print("x neutral", x_neutral)
#     return torch.sum(x_neutral)
#     # return 0

def optimize_NRE(x: torch.tensor, y, radius, neutral=0, lr=0.1):
    """
    :param x: (n_pts, n_feature_maps, n_dim)
    :param y:
    :param neutral:
    :return:
    """
    n_dim = x.shape[-1]
    n_feat_maps = x.shape[1]

    # Initialize the shift to zero
    shift = torch.full((n_feat_maps, n_dim), 1, dtype=torch.float64 ,requires_grad=True)
    # Initialize the neutral-radius to zero
    radius = torch.full([n_feat_maps], 0.01, dtype=torch.float64 ,requires_grad=True)
    parameters = [shift, radius]

    # for i in range(20):
    for i in range(1):
        x_shifted = x - shift
        print("shape x_shifted", x_shifted.shape)

        tun_vectors = compute_tun_vectors(x_shifted, y, neutral)
        print("tun_vectors", tun_vectors.shape)
        print(tun_vectors)

        # get projections
        projections = compute_projections(x_shifted, tun_vectors)

        # compute loss
        loss = compute_loss(projections, x, y, radius)

        # plot_space(x_shifted, y, tun_vectors=tun_vectors[:, 0], save=True, name=str(i))
        plot_space(x_shifted, y, tun_vectors=tun_vectors[:, 0], radius=radius)
        print("shift", shift)
        print("radius", radius)
        loss.backward()
        print("shift grad", shift.grad)
        print("radius grad", radius.grad)

        # print("shift grad", shift.grad)
        with torch.no_grad():
            shift = shift - lr * shift.grad
            shift.requires_grad = True
            radius = radius - lr * radius.grad
            radius.requires_grad = True



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
    x_train, y_train = torch.tensor(x_train).unsqueeze(1), torch.tensor(y_train)
    x_train.requires_grad = True
    optimize_NRE(x_train, y_train, radius=1)
