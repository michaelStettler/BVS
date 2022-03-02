import numpy as np

N_FILTERS = 9


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

    small_shift_left = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                                 [-1., -1., -.5, -1., -1., -1., -1.],
                                 [-1., -.5, 0.0, 0.0, -.5, -1., -1.],
                                 [-.5, 0.0, 1.0, 1.0, -.5, -.5, -1.],
                                 [-1., -.5, 1.0, 1.0, 1.0, -.5, -1.],
                                 [-1., -.5, 1.0, 1.0, 1.0, 1.0, -.5],
                                 [-1., -1., -.5, 1.0, 1.0, 1.0, -.5]]) / 12

    # idx 16 -> start "end" primer
    custom1 = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                        [-1., -1., -1., -1., -1., -1., -1.],
                        [-1., -1., -1., -1., -1., -1., -1.],
                        [-1., -1., -1., 0.03, 0.03, -1., -1.],
                        [-1., -1., 0.3, 0.9, 0.3, -1., -1.],
                        [-1., -1., 0.7, 1.0, 0.3, -1., -1.],
                        [-1., 0.5, 0.9, 0.8, -1., -1., -1.]]) / 5.76
    # idx 16 -> end "end" primer
    custom2 = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                        [-1., -1., 0.153, 0.627, -1., -1., -1.],
                        [-1., -1., 0.486, 0.945, 0.125, -1., -1.],
                        [-1., -1., 0.263, 0.996, 0.204, -1., -1.],
                        [-1., -1., 0.078, 0.882, 0.498, -1., -1.],
                        [-1., -1., -1., 0.812, 0.875, -1., -1.],
                        [-1., -1., -1., 0.616, 0.875, -1., -1.]]) / 8.435

    #idx 17 -> start "end" primer
    custom3 = np.array([[-1., -1., -1., -1., -1., -1., -1.],
                        [-1., 0.157, 0.067, 0., -1., -1., -1.],
                        [-2., 0.98, 0.561, 0., -1., -1., -1.],
                        [-2., 0.663, 0.867, 0.039, -2., -1., -1.],
                        [-2., 0.133, 0.949, 0.561, 0.008, -2., -1.],
                        [-1., -2., 0.227, 0.969, 0.722, 0.055, -1.],
                        [-1., -1., -2., 0.294, 0.961, 0.784, 0.11]]) / 9.107
    # idx 18 right end
    # custom4 = np.array([[-1., -2.  , -2. , -2.  , 0.   , 0.   , 0.0],
    #                     [-1., 0.051, 0.22, 0.024, -2.  , -1.  , -1.],
    #                     [-1., 0.075, 1.  , 1.   , 0.376, -2.  , -1.],
    #                     [-1., -2.  , -2. , 1.   , 1.   , -2.  , -1.],
    #                     [-1., -2.  , -2. , 0.051, 1.   , 0.251, -2.],
    #                     [-1., -2.  , -2. , 0.341, 1.   , 0.169, -2.],
    #                     [0. , 0.   , 0.  , 0.   , 0.   , 0.   , 0.]]) / 6.5

    custom4 = np.array([[-2., -3., -1., -1., 0., 0., 0.0],
                        [-2., 0.051, 0.22, 0.024, -1., -1., -1.],
                        [-1., 0.075, 1., 1., 0.376, -1., -1.],
                        [-1., -1., -1., 1., 1., -2., -1.],
                        [-1., -1., -1., 0.051, 1., 0.251, -1.],
                        [-1., -1., -1., 0.341, 1., 0.169, -1.],
                        [0., 0., 0., 0., 0., 0., 0.]]) / 6.5

    # top_end = np.array([top_end_large, top_end_medium])
    filters.append(large)
    filters.append(medium)
    filters.append(medium_shift_left)
    filters.append(medium_shift_right)
    filters.append(small_shift_left)
    filters.append(custom1)
    filters.append(custom2)
    filters.append(custom3)
    filters.append(custom4)

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
    # idx 17 get type 11 (left-top) corner of the 8
    custom1 = np.array([[-5., 0.125, 0.051, 0.075, -5., -5., -5.],
                        [-5., 0.898, 0.843, 0.737, 0.361, 0.031, -5.],
                        [-5., 0.796, 0.996, 1., 0.996, 0.914, 0.416],
                        [-5., 0.043, 0.373, 0.686, 0.992, 0.992, 0.957],
                        [-5., -1., 0.118, 0.522, 0.992, 0.914, 0.996],
                        [-5., -1., 0.184, 0.992, 0.894, 0.188, 0.667],
                        [-5., -1., 0.184, 0.992, 0.631, -10., 0.141]]) / 20.7

    # add all filters
    filters.append(large)
    filters.append(medium)
    filters.append(medium_shift_left)
    filters.append(medium_shift_right)
    filters.append(small_shift_left)
    filters.append(custom1)

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
                      [0.0, -.5, -.5, -.5, -.5, -.5, 0.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      [-2., -3., 1.0, 1.0, 1.0, -3., -2.],
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

    medium = np.array([[1.0, 0.0, -1., -1., -1., 0.0, 1.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [-1., 0.0, 1.0, 0.0, 1.0, 0.0, -1.],
                       [-1., 0.0, 0.0, 1.0, 0.0, 0.0, -1.],
                       [-1., 0.0, 1.0, 0.0, 1.0, 0.0, -1.],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [1.0, 0.0, -1., -1., -1., 0.0, 1.0]]) / 12

    medium_shift_left = np.array([[1.0, 0.0, 0.0, -1.5, 0.0, 0.0, 1.0],
                                  [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                  [-1.5, 0.0, 1.0, 1.0, 1.0, 0.0, -1.5],
                                  [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                  [1.0, 0.0, 0.0, -1.5, 0.0, 0.0, 1.0]]) / 17

    medium_shift_right = np.array([[0.0, 0.0, -2., -2., -1., 0.0, 1.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, -2.],
                                   [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                   [-2., 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 0.0, -1., -2., -2., 0.0, 0.0]]) / 15

    custom1 = np.array([[-5., -5., 0.027, 0.843, 0.996, 0.502, -5.],
                        [-5., -5., 0.153, 0.996, 0.996, 0.22, -5.],
                        [0.027, 0.137, 0.384, 0.996, 0.996, 0.816, 0.616],
                        [0.831, 0.996, 0.996, 0.996, 0.996, 0.984, 0.875],
                        [0.894, 0.98, 0.996, 0.996, 0.604, 0.196, -5.],
                        [-5., 0.847, 0.996, 0.863, 0.047, -5., -5.],
                        [0.537, 0.957, 0.91, 0.196, -5., -5., -5.]]) / 25.398

    # add all filters
    filters.append(large)
    filters.append(medium)
    filters.append(medium_shift_left)
    filters.append(medium_shift_right)
    filters.append(custom1)

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
