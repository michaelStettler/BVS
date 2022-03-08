import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def build_primer(lmk_pos, thresh_val=0.75, filt_size=(7, 7)):
    """

    :param lmk_pos: nx4 matrix, with 4 the following parameters: 0: idx of the mnsit img, 1: posX, 2: posX, 3: n_rotate
    :param thresh_val:
    :param filt_size:
    :return:
    """
    pad_x = int(filt_size[0] / 2)
    pad_y = int(filt_size[1] / 2)

    # define filter for primer
    patch = x_train[lmk_pos[0], lmk_pos[1]-pad_x:lmk_pos[1]+pad_x+1, lmk_pos[2]-pad_y:lmk_pos[2]+pad_y+1] / 255.
    filter = np.copy(patch)

    # control size
    if np.shape(filter)[0] != filt_size[0]:
        print("Dimension 0 of patch is not matching ,expected {}, received {}".format(filt_size[0], np.shape(patch)[0]))
    if np.shape(filter)[1] != filt_size[1]:
        print("Dimension 1 of patch is not matching ,expected {}, received {}".format(filt_size[1], np.shape(patch)[1]))

    # compute the number of zeros within the patch
    n_zeros = np.count_nonzero(patch == 0)

    # compute alpha factor to reach the thresh_value
    max_matching_val = np.sum(filter * patch)
    alpha = thresh_val / max_matching_val

    # normalize filter
    filter *= alpha

    # set filter to neutral with the zeros
    offset_neg_val = -5 * np.sum(filter) / n_zeros
    filter[filter == 0] = offset_neg_val

    # apply number of rotation
    filter = np.rot90(filter, lmk_pos[3])

    return filter


def get_top_primer_multi_scale(lmks_pos, n_filters, filt_size=(7, 7)):
    filters = []

    # create filters
    for lmk_pos in lmks_pos:
        custom_filt = build_primer(lmk_pos, filt_size=filt_size)
        filters.append(custom_filt)

    # add zeros if not equal to N_FILTERS
    if len(filters) < n_filters:
        for i in range(n_filters - len(filters)):
            filters.append(np.zeros((filt_size[0], filt_size[1])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_ends_filters_multi_scale(lmks_pos, n_filters, filt_size=(7, 7)):
    top_end = get_top_primer_multi_scale(lmks_pos, n_filters, filt_size=filt_size)
    right_end = np.rot90(top_end, 1, axes=(1, 2))
    down_end = np.rot90(top_end, 2, axes=(1, 2))
    left_end = np.rot90(top_end, 3, axes=(1, 2))

    return np.concatenate([top_end, right_end, down_end, left_end])


def get_top_right_corner_multi_scale(lmks_pos, n_filters, filt_size=(7, 7)):
    filters = []

    # create filters
    for lmk_pos in lmks_pos:
        custom_filt = build_primer(lmk_pos, filt_size=filt_size)
        filters.append(custom_filt)

    # add zeros if not equal to N_FILTERS
    if len(filters) < n_filters:
        for i in range(n_filters - len(filters)):
            filters.append(np.zeros((filt_size[0], filt_size[1])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_corners_filters_multi_scale(lmks_pos, n_filters, filt_size=(7, 7)):
    top_right = get_top_right_corner_multi_scale(lmks_pos, n_filters, filt_size=filt_size)
    down_right = np.rot90(top_right, 1, axes=(1, 2))
    down_left = np.rot90(top_right, 2, axes=(1, 2))
    top_left = np.rot90(top_right, 3, axes=(1, 2))

    return np.concatenate([top_right, down_right, down_left, top_left])


def get_top_T_multi_scale(lmks_pos, n_filters, filt_size=(7, 7)):
    filters = []

    # create filters
    for lmk_pos in lmks_pos:
        custom_filt = build_primer(lmk_pos, filt_size=filt_size)
        filters.append(custom_filt)

    # add zeros if not equal to N_FILTERS
    if len(filters) < n_filters:
        for i in range(n_filters - len(filters)):
            filters.append(np.zeros((filt_size[0], filt_size[0])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_T_filters_multi_scale(lmks_pos, n_filters, filt_size=(7, 7)):
    top_T = get_top_T_multi_scale(lmks_pos, n_filters, filt_size=filt_size)
    right_T = np.rot90(top_T, 1, axes=(1, 2))
    down_T = np.rot90(top_T, 2, axes=(1, 2))
    left_T = np.rot90(top_T, 3, axes=(1, 2))

    return np.concatenate([top_T, right_T, down_T, left_T])


def get_cross_multi_scale(lmks_pos, n_filters, filt_size=(7, 7)):
    filters = []

    # create filters
    for lmk_pos in lmks_pos:
        custom_filt = build_primer(lmk_pos, filt_size=filt_size)
        filters.append(custom_filt)

    # add zeros if not equal to N_FILTERS
    if len(filters) < n_filters:
        for i in range(n_filters - len(filters)):
            filters.append(np.zeros((filt_size[0], filt_size[0])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_filters_multi_scale(lmks_pos, filt_size=(7, 7)):
    n_filters = 0

    # get higher number of filters
    for lmk_pos in lmks_pos:
        n_filt = len(lmk_pos)
        if n_filt > n_filters:
            n_filters = n_filt

    print("max filters:", n_filters)

    ends_filters = get_ends_filters_multi_scale(lmks_pos[0], n_filters, filt_size=filt_size)

    T_filters = get_T_filters_multi_scale(lmks_pos[1], n_filters, filt_size=filt_size)

    corners_filters = get_corners_filters_multi_scale(lmks_pos[2], n_filters, filt_size=filt_size)

    cross_filters = get_cross_multi_scale(lmks_pos[3], n_filters, filt_size=filt_size)

    print("shape ends_filters", np.shape(ends_filters))
    print("shape corners_filters", np.shape(corners_filters))
    print("shape T_filters", np.shape(T_filters))
    print("shape cross_filters", np.shape(cross_filters))

    return np.concatenate([ends_filters, T_filters, corners_filters, cross_filters])
