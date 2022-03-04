import numpy as np
import tensorflow as tf

shape_filters = (7, 7)
N_FILTERS = 11
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def build_primer(patch, n_rotate, thresh_val=0.75, filt_size=(7, 7)):
    # control size
    if np.shape(patch)[0] != filt_size[0]:
        print("Dimension 0 of patch is not matching ,expected {}, received {}".format(filt_size[0], np.shape(patch)[0]))
    if np.shape(patch)[1] != filt_size[1]:
        print("Dimension 1 of patch is not matching ,expected {}, received {}".format(filt_size[1], np.shape(patch)[1]))

    # define filter for primer
    filter = np.copy(patch)

    # compute the number of zeros within the patch
    n_zeros = np.count_nonzero(patch==0)

    # compute alpha factor to reach the thresh_value
    max_matching_val = np.sum(filter * patch)
    alpha = thresh_val / max_matching_val

    # normalize filter
    filter *= alpha

    # set filter to neutral with the zeros
    offset_neg_val = -5 * np.sum(filter) / n_zeros
    filter[filter == 0] = offset_neg_val

    # apply number of rotation
    filter = np.rot90(filter, n_rotate)

    return filter


def get_top_primer_multi_scale():
    filters = []

    # create filters
    custom1 = build_primer(x_train[0, 5:12, 5:12] / 255., 3)
    custom2 = build_primer(x_train[0, 3:10, 19:26] / 255., 1)
    custom3 = build_primer(x_train[0, 20:27, 3:10] / 255., 3)
    custom4 = build_primer(x_train[2, 4:11, 2:9] / 255., 0)
    custom5 = build_primer(x_train[3, 4:11, 16:23] / 255., 0)
    custom6 = build_primer(x_train[4, 21:28, 13:20] / 255., 2)

    # add all filters
    filters.append(custom1)
    filters.append(custom2)
    filters.append(custom3)
    filters.append(custom4)
    filters.append(custom5)
    filters.append(custom6)

    # add zeros if not equal to N_FILTERS
    if len(filters) < N_FILTERS:
        for i in range(N_FILTERS - len(filters)):
            filters.append(np.zeros((shape_filters[0], shape_filters[1])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_right_primer_multi_scale():
    return np.rot90(get_top_primer_multi_scale(), 1, axes=(1, 2))


def get_down_primer_multi_scale():
    return np.rot90(get_top_primer_multi_scale(), 2, axes=(1, 2))


def get_left_primer_multi_scale():
    return np.rot90(get_top_primer_multi_scale(), 3, axes=(1, 2))


def get_ends_filters_multi_scale():
    top_end = get_top_primer_multi_scale()
    right_end = get_right_primer_multi_scale()
    down_end = get_down_primer_multi_scale()
    left_end = get_left_primer_multi_scale()

    return np.concatenate([top_end, right_end, down_end, left_end])


def get_top_right_corner_multi_scale():
    filters = []

    # create filters
    custom1 = build_primer(x_train[0, 9:16, 10:17] / 255., 3)
    custom2 = build_primer(x_train[0, 12:19, 15:22] / 255., 1)
    custom3 = build_primer(x_train[0, 16:23, 15:22] / 255., 2)
    custom4 = build_primer(x_train[1, 12:19, 5:12] / 255., 0)
    custom5 = build_primer(x_train[1, 18:25, 5:12] / 255., 3)
    custom6 = build_primer(x_train[1, 12:19, 17:24] / 255., 2)
    custom7 = build_primer(x_train[1, 3:10, 16:23] / 255., 1)
    custom8 = build_primer(x_train[2, 12:19, 2:9] / 255., 3)
    custom9 = build_primer(x_train[4, 6:13, 9:16] / 255., 0)
    custom10 = build_primer(x_train[4, 11:18, 5:12] / 255., 3)
    custom11 = build_primer(x_train[4, 6:13, 15:22] / 255., 1)
    # customX = build_primer(x_train[5, 12:19, 10:17] / 255., 0)

    # add all filters
    filters.append(custom1)
    filters.append(custom2)
    filters.append(custom3)
    filters.append(custom4)
    filters.append(custom5)
    filters.append(custom6)
    filters.append(custom7)
    filters.append(custom8)
    filters.append(custom9)
    filters.append(custom10)
    filters.append(custom11)

    # add zeros if not equal to N_FILTERS
    if len(filters) < N_FILTERS:
        for i in range(N_FILTERS - len(filters)):
            filters.append(np.zeros((shape_filters[0], shape_filters[1])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters


def get_down_right_corner_multi_scale():
    return np.rot90(get_top_right_corner_multi_scale(), 1, axes=(1, 2))


def get_down_left_corner_multi_scale():
    return np.rot90(get_top_right_corner_multi_scale(), 2, axes=(1, 2))


def get_top_left_corner_multi_scale():
    return np.rot90(get_top_right_corner_multi_scale(), 3, axes=(1, 2))


def get_corners_filters_multi_scale():
    top_right = get_top_right_corner_multi_scale()
    down_right = get_down_right_corner_multi_scale()
    down_left = get_down_left_corner_multi_scale()
    top_left = get_top_left_corner_multi_scale()

    return np.concatenate([top_right, down_right, down_left, top_left])


def get_top_T_multi_scale():
    filters = []

    # create filters
    custom1 = build_primer(x_train[0, 5:12, 9:16] / 255., 0)
    custom2 = build_primer(x_train[2, 11:18, 15:22] / 255., 1)
    custom3 = build_primer(x_train[4, 12:19, 12:19] / 255., 1)

    # add all filters
    filters.append(custom1)
    filters.append(custom2)
    filters.append(custom3)

    # add zeros if not equal to N_FILTERS
    if len(filters) < N_FILTERS:
        for i in range(N_FILTERS - len(filters)):
            filters.append(np.zeros((shape_filters[0], shape_filters[0])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters



def get_right_T_multi_scale():
    return np.rot90(get_top_T_multi_scale(), 1, axes=(1, 2))


def get_down_T_multi_scale():
    return np.rot90(get_top_T_multi_scale(), 2, axes=(1, 2))


def get_left_T_multi_scale():
    return np.rot90(get_top_T_multi_scale(), 3, axes=(1, 2))


def get_cross_multi_scale():
    filters = []

    # create filters

    # add all filters

    # add zeros if not equal to N_FILTERS
    if len(filters) < N_FILTERS:
        for i in range(N_FILTERS - len(filters)):
            filters.append(np.zeros((shape_filters[0], shape_filters[0])))

    # re order axis
    filters = np.moveaxis(filters, 0, -1)
    filters = np.expand_dims(filters, axis=0)
    return filters



def get_T_filters_multi_scale():
    top_T = get_top_T_multi_scale()
    right_T = get_right_T_multi_scale()
    down_T = get_down_T_multi_scale()
    left_T = get_left_T_multi_scale()

    return np.concatenate([top_T, right_T, down_T, left_T])


def get_filters_multi_scale():
    ends_filters = get_ends_filters_multi_scale()

    corners_filters = get_corners_filters_multi_scale()

    T_filters = get_T_filters_multi_scale()

    cross_filters = get_cross_multi_scale()

    print("shape ends_filters", np.shape(ends_filters))
    print("shape corners_filters", np.shape(corners_filters))
    print("shape T_filters", np.shape(T_filters))
    print("shape cross_filters", np.shape(cross_filters))

    return np.concatenate([ends_filters, T_filters, corners_filters, cross_filters])
