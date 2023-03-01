import numpy as np


def generate_data_set(n_dim: int, n_cat: int, n_points: int, min_length=2, max_length=7, ref_at_origin=True,
                      n_latent=1, n_ref=1, variance_ratio=1, ref_variance=1, balanced=True):
    """

    :param n_dim:
    :param n_cat:
    :param n_points: n_points per category
    :param ref_at_0:
    :param balanced:
    :return:
    """

    if max_length < min_length:
        max_length = min_length

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
            ref_pos = np.random.rand(n_points, n_dim) * variance_ratio + ref_origin

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