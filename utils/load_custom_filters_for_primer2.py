import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def simple_ones_patch_wh_neg_value(filter, patch, thresh_activation, neg_factor):
    # compute the number of zeros within the patch
    n_zeros = np.count_nonzero(patch == 0)

    # set filter values to 0 and 1
    filter[filter > 0.5] = 1.0
    filter[filter < 0.5] = 0.0

    # compute alpha factor to reach the thresh_value
    max_matching_val = np.sum(filter * patch)
    alpha = thresh_activation / max_matching_val

    # normalize filter
    filter *= alpha

    # set filter to neutral with the zeros
    offset_neg_val = - neg_factor * np.sum(filter) / n_zeros
    filter[patch == 0] = offset_neg_val

    return filter


def simple_patch_wh_neg_value(filter, patch, thresh_activation, neg_factor):
    # compute the number of zeros within the patch
    n_zeros = np.count_nonzero(patch == 0)

    # compute alpha factor to reach the thresh_value
    max_matching_val = np.sum(filter * patch)
    alpha = thresh_activation / max_matching_val

    # normalize filter
    filter *= alpha

    # set filter to neutral with the zeros
    offset_neg_val = - neg_factor * np.sum(filter) / n_zeros
    filter[filter == 0] = offset_neg_val

    return filter


def build_primer(lmk_pos, thresh_val=0.75, filt_size=(7, 7), neg_factor=6):
    """

    :param lmk_pos: nx4 matrix, with 4 the following parameters: 0: idx of the mnsit img, 1: posX, 2: posX, 3: n_rotate
    :param thresh_val:
    :param filt_size:
    :param neg_factor: value to which the negative parts of the filters get multiplied
    :return:
    """
    pad_x = int(filt_size[0] / 2)
    pad_y = int(filt_size[1] / 2)

    # define start and stop
    start_x = lmk_pos[1]-pad_x
    end_x = lmk_pos[1]+pad_x+1
    start_y = lmk_pos[2]-pad_y
    end_y = lmk_pos[2]+pad_y+1

    # define filter for primer
    img = x_train[lmk_pos[0]] / 255.
    patch = img[np.maximum(start_x, 0):end_x, np.maximum(start_y, 0):end_y]

    # add zeros if patch is smaller than the dimension
    if start_x < 0:
        zeros_patch = np.zeros((filt_size[0] - np.shape(patch)[0], np.shape(patch)[1]))
        patch = np.vstack([zeros_patch, patch])
    elif end_x > np.shape(img)[0]:
        zeros_patch = np.zeros((end_x - np.shape(img)[0], np.shape(patch)[1]))
        patch = np.vstack([patch, zeros_patch])
    if start_y < 0:
        zeros_patch = np.zeros((np.shape(patch)[0], filt_size[1] - np.shape(patch)[1]))
        patch = np.hstack([zeros_patch, patch])
    elif end_y > np.shape(img)[1]:
        zeros_patch = np.zeros((np.shape(patch)[0], end_y - np.shape(img)[1]))
        patch = np.hstack([patch, zeros_patch])

    # create filter
    filter = np.copy(patch)

    # control size
    if np.shape(filter)[0] != filt_size[0]:
        print("Dimension 0 of patch is not matching ,expected {}, received {}".format(filt_size[0], np.shape(patch)[0]))
    if np.shape(filter)[1] != filt_size[1]:
        print("Dimension 1 of patch is not matching ,expected {}, received {}".format(filt_size[1], np.shape(patch)[1]))

    # build filter
    filter = simple_patch_wh_neg_value(filter, patch, thresh_val, neg_factor)
    # filter = simple_ones_patch_wh_neg_value(filter, patch, thresh_val, neg_factor)

    # apply number of rotation
    filter = np.rot90(filter, lmk_pos[3])

    return filter


def get_top_primer_multi_scale(lmks_pos, n_filters, filt_size=(7, 7), thresh_val=0.75, neg_factor=6):
    filters = []

    # create filters
    for lmk_pos in lmks_pos:
        custom_filt = build_primer(lmk_pos, filt_size=filt_size,
                                   thresh_val=thresh_val,
                                   neg_factor=neg_factor)
        filters.append(custom_filt)

    # add zeros if not equal to N_FILTERS
    if len(filters) < n_filters:
        for i in range(n_filters - len(filters)):
            filters.append(np.zeros((filt_size[0], filt_size[1])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_ends_filters_multi_scale(lmks_pos, n_filters, filt_size=(7, 7), thresh_val=0.75, neg_factor=6):
    top_end = get_top_primer_multi_scale(lmks_pos, n_filters,
                                         filt_size=filt_size,
                                         thresh_val=thresh_val,
                                         neg_factor=neg_factor)
    right_end = np.rot90(top_end, 1, axes=(1, 2))
    down_end = np.rot90(top_end, 2, axes=(1, 2))
    left_end = np.rot90(top_end, 3, axes=(1, 2))

    return np.concatenate([top_end, right_end, down_end, left_end])


def get_top_right_corner_multi_scale(lmks_pos, n_filters, filt_size=(7, 7), thresh_val=0.75, neg_factor=6):
    filters = []

    # create filters
    for lmk_pos in lmks_pos:
        custom_filt = build_primer(lmk_pos, filt_size=filt_size,
                                   thresh_val=thresh_val,
                                   neg_factor=neg_factor)
        filters.append(custom_filt)

    # add zeros if not equal to N_FILTERS
    if len(filters) < n_filters:
        for i in range(n_filters - len(filters)):
            filters.append(np.zeros((filt_size[0], filt_size[1])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_corners_filters_multi_scale(lmks_pos, n_filters, filt_size=(7, 7), thresh_val=0.75, neg_factor=6):
    top_right = get_top_right_corner_multi_scale(lmks_pos, n_filters,
                                                 filt_size=filt_size,
                                                 thresh_val=thresh_val,
                                                 neg_factor=neg_factor)
    down_right = np.rot90(top_right, 1, axes=(1, 2))
    down_left = np.rot90(top_right, 2, axes=(1, 2))
    top_left = np.rot90(top_right, 3, axes=(1, 2))

    return np.concatenate([top_right, down_right, down_left, top_left])


def get_top_T_multi_scale(lmks_pos, n_filters, filt_size=(7, 7), thresh_val=0.75, neg_factor=6):
    filters = []

    # create filters
    for lmk_pos in lmks_pos:
        custom_filt = build_primer(lmk_pos, filt_size=filt_size,
                                   thresh_val=thresh_val,
                                   neg_factor=neg_factor)
        filters.append(custom_filt)

    # add zeros if not equal to N_FILTERS
    if len(filters) < n_filters:
        for i in range(n_filters - len(filters)):
            filters.append(np.zeros((filt_size[0], filt_size[0])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_T_filters_multi_scale(lmks_pos, n_filters, filt_size=(7, 7), thresh_val=0.75, neg_factor=6):
    top_T = get_top_T_multi_scale(lmks_pos, n_filters,
                                  filt_size=filt_size,
                                  thresh_val=thresh_val,
                                  neg_factor=neg_factor)
    right_T = np.rot90(top_T, 1, axes=(1, 2))
    down_T = np.rot90(top_T, 2, axes=(1, 2))
    left_T = np.rot90(top_T, 3, axes=(1, 2))

    return np.concatenate([top_T, right_T, down_T, left_T])


def get_cross_multi_scale(lmks_pos, n_filters, filt_size=(7, 7), thresh_val=0.75, neg_factor=6):
    filters = []

    # create filters
    for lmk_pos in lmks_pos:
        custom_filt = build_primer(lmk_pos, filt_size=filt_size,
                                   thresh_val=thresh_val,
                                   neg_factor=neg_factor)
        filters.append(custom_filt)

    # add zeros if not equal to N_FILTERS
    if len(filters) < n_filters:
        for i in range(n_filters - len(filters)):
            filters.append(np.zeros((filt_size[0], filt_size[0])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_filters_multi_scale(lmks, extend_by_type=True, filt_size=(7, 7), thresh_val=0.75, neg_factor=6):
    n_filters = 0

    # get higher number of filters
    for lmk_pos in lmks_pos:
        n_filt = len(lmk_pos)
        if n_filt > n_filters:
            n_filters = n_filt

    print("max filters:", n_filters)

    # declare filters
    n_type_filter = np.amax(lmks[:, 3])
    print(n_type_filter)
    # count max filters
    for lmk in lmks:

        filter = get_filters(lmk, n_filters,
                                                filt_size=filt_size,
                                                thresh_val=thresh_val,
                                                neg_factor=neg_factor)

        if extend_by_type:
            if lmk[3] in [0, 1, 2, 3]:
                filters[0].append(filter)


    T_filters = get_T_filters_multi_scale(lmks_pos[1], n_filters,
                                          filt_size=filt_size,
                                          thresh_val=thresh_val,
                                          neg_factor=neg_factor)

    corners_filters = get_corners_filters_multi_scale(lmks_pos[2], n_filters,
                                                      filt_size=filt_size,
                                                      thresh_val=thresh_val,
                                                      neg_factor=neg_factor)

    cross_filters = get_cross_multi_scale(lmks_pos[3], n_filters,
                                          filt_size=filt_size,
                                          thresh_val=thresh_val,
                                          neg_factor=neg_factor)

    print("shape ends_filters", np.shape(ends_filters))
    print("shape corners_filters", np.shape(corners_filters))
    print("shape T_filters", np.shape(T_filters))
    print("shape cross_filters", np.shape(cross_filters))

    return np.concatenate([ends_filters, T_filters, corners_filters, cross_filters])
