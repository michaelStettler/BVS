import numpy as np

N_FILTERS = 5


def get_top_primer():
    return np.array([[-1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., -1., -1., -1., -1., -1.],
                    [-1., -1., 1.0, 1.0, 1.0, -1., -1.],
                    [-1., -1., 1.0, 1.0, 1.0, -1., -1.],
                    [-1., -1., 1.0, 1.0, 1.0, -1., -1.],
                    [-1., 0.0, 1.0, 1.0, 1.0, 0.0, -1.]]) / 12


def get_right_primer():
    return np.rot90(get_top_primer(), 1)


def get_down_primer():
    return np.rot90(get_top_primer(), 2)


def get_left_primer():
    return np.rot90(get_top_primer(), 3)


def get_top_right_corner():
    return np.array([[-4., -4., -4., -4., -4., -4., -2.],
                     [-4., -4., -4., -4., -2., -2., 0.0],
                     [-4., -4., 1.0, 1.0, 1.0, 1.0, 1.0],
                     [-4., -4., 1.0, 1.0, 1.0, 1.0, 1.0],
                     [-4., -2., 1.0, 1.0, 1.0, 1.0, 1.0],
                     [-4., -2., 1.0, 1.0, 1.0, -1., 0.0],
                     [-2., 0.0, 1.0, 1.0, 1.0, 0.0, -2.]]) / 21


def get_down_right_corner():
    return np.rot90(get_top_right_corner(), 1)


def get_down_left_corner():
    return np.rot90(get_top_right_corner(), 2)


def get_top_left_corner():
    return np.rot90(get_top_right_corner(), 3)


def get_top_T():
    return np.array([[-4., -4., -4., -4., -4., -4., -4.],
                     [-4., -4., -4., -4., -4., -4., -4.],
                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                     [-2., -4., 1.0, 1.0, 1.0, -4., -2.],
                     [-6., 0.0, 1.0, 1.0, 1.0, 0.0, -6.]]) / 27


def get_right_T():
    return np.rot90(get_top_T(), 1)


def get_down_T():
    return np.rot90(get_top_T(), 2)


def get_left_T():
    return np.rot90(get_top_T(), 3)


def get_ends_filters():
    top = get_top_primer()
    right = get_right_primer()
    down = get_down_primer()
    left = get_left_primer()

    return np.array([top, right, down, left])


def get_corners_filters():
    top_right = get_top_right_corner()
    down_right = get_down_right_corner()
    down_left = get_down_left_corner()
    top_left = get_top_left_corner()

    return np.array([top_right, down_right, down_left, top_left])


def get_T_filters():
    top_T = get_top_T()
    right_T = get_right_T()
    down_T = get_down_T()
    left_T = get_left_T()

    return np.array([top_T, right_T, down_T, left_T])


def get_filters():
    ends_filters = get_ends_filters()
    corners_filters = get_corners_filters()
    T_filters = get_T_filters()

    return np.concatenate([ends_filters, T_filters, corners_filters])


def get_top_primer_multi_scale():
    filters = []
    large = np.array([[-2., -2., -2., -2., -2., -2., -2.],
                      [-2., -2., -2., -2., -1., -2., -2.],
                      [-2., -.5, -.5, -.5, -.5, -.5, -2.],
                      [-2., -.5, 1.0, 1.0, 1.0, -.5, -2.],
                      [-2., 0.0, 1.0, 1.0, 1.0, 0.0, -2.],
                      [-2., 0.0, 1.0, 1.0, 1.0, 0.0, -2.],
                      [-2., 0.0, 1.0, 1.0, 1.0, 0.0, -2.]]) / 12

    medium = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                       [-1., -1., -.5, -.5, -1., -1., -1.],
                       [-1., -.5, 0.0, 0.0, -.5, -1., -1.],
                       [-1., 0.0, 1.0, 1.0, 0.0, -1., -1.],
                       [-1., 0.0, 1.0, 1.0, 0.0, -1., -1.],
                       [-.5, 0.0, 1.0, 1.0, 0.0, -.5, -1.],
                       [-.5, 0.0, 1.0, 1.0, 0.0, -.5, -1.]]) / 8

    medium_shift_left = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                                  [-1., -1., -1., -1., -1., -1., -1.],
                                  [-1., -.5, -.5, -.5, -.5, -1., -1.],
                                  [-1., -.5, 1.0, 1.0, -.5, -.5, -1.],
                                  [-1., -.5, -.5, 1.0, 1.0, -.5, -1.],
                                  [-1., -1., -.5, 1.0, 1.0, -.5, -1.],
                                  [-1., -1., -.5, 0.0, 1.0, 1.0, -.5]]) / 8

    medium_shift_right = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                                   [-1., -1., -1., -1., -1., -1., -1.],
                                   [-1., -1., -.5, -.5, -.5, -.5, -1.],
                                   [-1., -.5, -.5, 1.0, 1.0, -.5, -1.],
                                   [-1., -.5, 1.0, 1.0, -.5, -.5, -1.],
                                   [-1., -.5, 1.0, 1.0, -.5, -1., -1.],
                                   [-.5, 1.0, 1.0, 0.0, -.5, -1.0, -1.]]) / 8

    # top_end = np.array([top_end_large, top_end_medium])
    filters.append(large)
    filters.append(medium)
    filters.append(medium_shift_left)
    filters.append(medium_shift_right)

    # add zeros if not equal to N_FILTERS
    if len(filters) < N_FILTERS:
        for i in range(N_FILTERS - len(filters)):
            filters.append(np.zeros((np.shape(filters)[1], np.shape(filters)[2])))

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
    large = np.array([[-4., -4., -4., -4., -4., -4., -2.],
                      [-4., -4., -4., -4., -2., -2., 0.0],
                      [-4., -4., 1.0, 1.0, 1.0, 1.0, 1.0],
                      [-4., -4., 1.0, 1.0, 1.0, 1.0, 1.0],
                      [-4., -2., 1.0, 1.0, 1.0, 1.0, 1.0],
                      [-4., -2., 1.0, 1.0, 1.0, -1., 0.0],
                      [-2., 0.0, 1.0, 1.0, 1.0, 0.0, -2.]]) / 21

    medium = np.array([[-2., -2., -2., -2., -2., -2., -2.],
                       [-2., -1., -1., -.5, -.5, -.5, 0.0],
                       [-2., -1., 1.0, 1.0, 1.0, 1.0, 0.5],
                       [-2., -.5, 1.0, 1.0, 1.0, 1.0, 0.5],
                       [-2., -.5, 1.0, 1.0, -.5, -.5, 0.0],
                       [-2., 0.0, 1.0, 1.0, -.5, -1., 0.0],
                       [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]]) / 12

    # medium_shift_left = np.array([[-2., -2., -2., -2., -1., 1.0, 0.0],
    #                               [-2., -1., -1., -.5, 1.0, 1.0, 0.0],
    #                               [-2., -1., 0.0, 1.0, 1.0, 0.0, 0.0],
    #                               [-2., -.5, 1.0, 1.0, -.5, -1., 0.0],
    #                               [-1., -.5, 1.0, 1.0, -.5, -1., 0.0],
    #                               [-1., 0.0, 1.0, 1.0, 0.0, -1., 0.0],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) / 12

    medium_shift_left = np.array([[-2., -2., -2., -2., -1., 0.0, 1.0],
                                  [-2., -1., -1., -.5, 1.0, 1.0, 1.0],
                                  [-2., -1., 0.0, 1.0, 1.0, 1.0, 0.0],
                                  [-2., -.5, 1.0, 1.0, -.5, -1., 0.0],
                                  [-1., -.5, 1.0, 1.0, -.5, -1., 0.0],
                                  [-1., 0.0, 1.0, 1.0, 0.0, -1., 0.0],
                                  [-1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) / 14

    medium_shift_right = np.array([[-2., -2., -1., -1., -1., -.5, 0.0],
                                   [-1., -1., -1., -1., -.5, -.5, 0.0],
                                   [-1., -.5, 1.0, 1.0, 1.0, 1.0, 0.0],
                                   [-1., 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                   [-1., 1.0, 1.0, 0.0, -1., -1., 0.0],
                                   [-.5, 1.0, 1.0, -.5, -1., -1., 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) / 12

    small_shift_left = np.array([[-2., -2., -2., -2., -1., 0.0, 0.0],
                                 [-2., -1., -1., -1., 0.0, 1.0, 0.0],
                                 [-2., -1., 0.0, 1.0, 1.0, 0.0, 0.0],
                                 [-2., -.5, 1.0, 1.0, -.5, -1., 0.0],
                                 [-1., -.5, 1.0, 0.0, -.5, -1., 0.0],
                                 [-1., 0.0, 1.0, 0.0, 0.0, -1., 0.0],
                                 [-1., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) / 7

    # add all filters
    filters.append(large)
    filters.append(medium)
    filters.append(medium_shift_left)
    filters.append(medium_shift_right)
    filters.append(small_shift_left)

    # add zeros if not equal to N_FILTERS
    if len(filters) < N_FILTERS:
        for i in range(N_FILTERS - len(filters)):
            filters.append(np.zeros((np.shape(filters)[1], np.shape(filters)[2])))

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
    large = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                      [-1., -1., -1., -1., -1., -1., -1.],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [-2., -4., 1.0, 1.0, 1.0, -4., -2.],
                      [-6., 0.0, 1.0, 1.0, 1.0, 0.0, -6.]]) / 27

    medium = np.array([[-1., -1., -2., -2., -1., -1., 0.0],
                       [-.5, -.5, -1., -1., -.5, -.5, 0.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                       [-1., -2., 1.0, 1.0, -2., -1., 0.0],
                       [-1., -1., 1.0, 1.0, -1., -1., 0.0],
                       [-2., 0.0, 1.0, 1.0, 0.0, -2., 0.0]]) / 18

    medium_shift_left = np.array([[1.0, -.5, -1., -2., -1., -1., 0.0],
                                  [1.0, 1.0, -1., -1., -.5, -.5, 0.0],
                                  [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                  [-.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                  [-1., -2., 1.0, 1.0, -1., -1., 0.0],
                                  [-1., -1., 1.0, 1.0, -.5, -1., 0.0],
                                  [-2., -1., 0.0, 1.0, 1.0, -1., 0.0]]) / 18

    # add all filters
    filters.append(large)
    filters.append(medium)
    filters.append(medium_shift_left)

    # add zeros if not equal to N_FILTERS
    if len(filters) < N_FILTERS:
        for i in range(N_FILTERS - len(filters)):
            filters.append(np.zeros((np.shape(filters)[1], np.shape(filters)[2])))

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
    # cross primers
    large = np.array([[-1., 0.0, 0.0, 1.0, 0.0, 0.0, -1.],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [-1., 0.0, 0.0, 1.0, 0.0, 0.0, -1.]]) / 17

    medium = np.array([[1.0, 0.0, 0.0, -1.1, 0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                       [-1.1, 0.0, 0.0, 1.0, 0.0, 0.0, -1.1],
                       [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0, -1.1, 0.0, 0.0, 1.0]]) / 12

    # add all filters
    filters.append(large)
    filters.append(medium)

    # add zeros if not equal to N_FILTERS
    if len(filters) < N_FILTERS:
        for i in range(N_FILTERS - len(filters)):
            filters.append(np.zeros((np.shape(filters)[1], np.shape(filters)[2])))

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
