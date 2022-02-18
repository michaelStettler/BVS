import numpy as np


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
    top_end = np.array([large, medium, medium_shift_left, medium_shift_right])
    top_end = np.moveaxis(top_end, 0, -1)
    top_end = np.expand_dims(top_end, axis=0)
    return top_end


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
    # large = np.array([[-4., -4., -4., -4., -4., -4., -2.],
    #                   [-4., -4., -4., -4., -2., -2., 0.0],
    #                   [-4., -4., 1.0, 1.0, 1.0, 1.0, 1.0],
    #                   [-4., -4., 1.0, 1.0, 1.0, 1.0, 1.0],
    #                   [-4., -2., 1.0, 1.0, 1.0, 1.0, 1.0],
    #                   [-4., -2., 1.0, 1.0, 1.0, -1., 0.0],
    #                   [-2., 0.0, 1.0, 1.0, 1.0, 0.0, -2.]]) / 21
    #
    # medium = np.array([[-2., -2., -2., -2., -2., -2., -2.],
    #                    [-2., -1., -1., -.5, -.5, -.5, 0.0],
    #                    [-2., -1., 1.0, 1.0, 1.0, 1.0, 0.5],
    #                    [-2., -.5, 1.0, 1.0, 1.0, 1.0, 0.5],
    #                    [-2., -.5, 1.0, 1.0, -.5, -.5, 0.0],
    #                    [-2., -.5, 1.0, 1.0, -.5, -3., 0.0],
    #                    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]]) / 12
    #
    # medium_shift_left = np.array([[-2., -2., -2., -2., -1., 1.0, 0.0],
    #                               [-2., -1., -1., -.5, 1.0, 1.0, 0.0],
    #                               [-2., -1., 0.0, 1.0, 1.0, 0.0, 0.0],
    #                               [-2., -.5, 1.0, 1.0, -.5, -1., 0.0],
    #                               [-1., -.5, 1.0, 1.0, -.5, -1., 0.0],
    #                               [-1., 0.0, 1.0, 1.0, 0.0, -1., 0.0],
    #                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) / 12
    #
    # medium_shift_right = np.array([[-2., -2., -2., -2., -2., -.5, 0.0],
    #                                [-2., -1., -1., -1., -.5, -.5, 0.0],
    #                                [-1., -1., 1.0, 1.0, 1.0, 1.0, 0.0],
    #                                [-.5, -.5, 1.0, 1.0, 1.0, 1.0, 0.0],
    #                                [-.5, 1.0, 1.0, 0.0, -1., -1., 0.0],
    #                                [0.0, 1.0, 1.0, -.5, -1., -1., 0.0],
    #                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) / 12

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

    medium_shift_left = np.array([[-2., -2., -2., -2., -1., 1.0, 0.0],
                                  [-2., -1., -1., -.5, 1.0, 1.0, 0.0],
                                  [-2., -1., 0.0, 1.0, 1.0, 0.0, 0.0],
                                  [-2., -.5, 1.0, 1.0, -.5, -1., 0.0],
                                  [-1., -.5, 1.0, 1.0, -.5, -1., 0.0],
                                  [-1., 0.0, 1.0, 1.0, 0.0, -1., 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) / 12

    medium_shift_right = np.array([[-2., -2., -2., -2., -2., -.5, 0.0],
                                   [-2., -1., -1., -1., -.5, -.5, 0.0],
                                   [-1., -.5, 1.0, 1.0, 1.0, 1.0, 0.0],
                                   [-1., 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                                   [-.5, 1.0, 1.0, 0.0, -1., -1., 0.0],
                                   [-.5, 1.0, 1.0, -.5, -1., -1., 0.0],
                                   [0.0, 0.0, 0.0, 0.0, -1., -1., 0.0]]) / 12

    # top_right_corner = np.array([large, medium])
    top_right_corner = np.array([large, medium, medium_shift_left, medium_shift_right])
    top_right_corner = np.moveaxis(top_right_corner, 0, -1)
    top_right_corner = np.expand_dims(top_right_corner, axis=0)
    return top_right_corner


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

    medium_shift_right = np.zeros((7, 7))

    # top_T = np.array([large, medium])
    top_T = np.array([large, medium, medium_shift_left, medium_shift_right])
    top_T = np.moveaxis(top_T, 0, -1)
    top_T = np.expand_dims(top_T, axis=0)
    return top_T


def get_right_T_multi_scale():
    return np.rot90(get_top_T_multi_scale(), 1, axes=(1, 2))


def get_down_T_multi_scale():
    return np.rot90(get_top_T_multi_scale(), 2, axes=(1, 2))


def get_left_T_multi_scale():
    return np.rot90(get_top_T_multi_scale(), 3, axes=(1, 2))


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

    return np.concatenate([ends_filters, T_filters, corners_filters])
