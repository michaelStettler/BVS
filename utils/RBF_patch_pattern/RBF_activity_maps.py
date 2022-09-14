import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os


def compute_RBF_pattern_activity_maps(ft_maps, patterns, sigmas=1, lmk_idx=None, disable_tqdm=False):
    """
    The function "convolve" with a RBF kernel all the patterns within the given feature maps

    small sigma => more selective
    large sigma => more robust

    :param ft_maps: (n_images, ft_size, ft_size, n_ft_maps)
    :param patterns: (n_patterns, n_lmks, k_size, k_size, n_ft_maps)
    :param sigmas:
    :return: activity_map (n_images, n_patterns, n_lmks, ft_size, ft_size)
    """
    n_patterns = len(patterns)

    # declare activity map
    if lmk_idx is None:
        activity_map = np.zeros((np.shape(ft_maps)[0], n_patterns, np.shape(patterns)[1], np.shape(ft_maps)[1], np.shape(ft_maps)[2]))
    else:
        activity_map = np.zeros((np.shape(ft_maps)[0], n_patterns, 1, np.shape(ft_maps)[1], np.shape(ft_maps)[2]))

    # compute padding
    pad_x = int(np.shape(patterns)[2] / 2)
    pad_y = int(np.shape(patterns)[3] / 2)

    # pad feature maps
    padded_ft_maps = np.zeros((np.shape(ft_maps)[0], np.shape(ft_maps)[1] + 2 * pad_x, np.shape(ft_maps)[2] + 2 * pad_y, np.shape(ft_maps)[3]))

    # compensate for padding of zeros
    if pad_x == 0 and pad_y == 0:
        padded_ft_maps = ft_maps
    elif pad_x == 0 and pad_y != 0:
        padded_ft_maps[:, :, pad_y:-pad_y, :] = ft_maps
    elif pad_x != 0 and pad_y == 0:
        padded_ft_maps[:, pad_x:-pad_x, :, :] = ft_maps
    else:
        padded_ft_maps[:, pad_x:-pad_x, pad_y:-pad_y, :] = ft_maps

    if lmk_idx is None:
        n_lmk = np.shape(patterns)[1]
        lmk_indexes = range(n_lmk)
    else:
        lmk_indexes = lmk_idx

    for l, l_idx in tqdm(enumerate(lmk_indexes), disable=disable_tqdm):

        if isinstance(sigmas, int):
            sigma = sigmas
        elif isinstance(sigmas, float):
            sigma = sigmas
        elif isinstance(sigmas, np.int32):
            sigma = sigmas
        elif isinstance(sigmas, np.int64):
            sigma = sigmas
        elif isinstance(sigmas, list):
            sigma = sigmas[l]
        elif isinstance(sigmas, np.ndarray):
            sigma = sigmas[l]
        elif isinstance(sigmas, tuple):
            sigma = sigmas[l]
        else:
            raise NotImplementedError("sigmas of type {} is not implemented yet".format(type(sigmas)))

        # expand pattern to match dimension of patch (n_images, n_pattern, 1 (lmk), k_size, k_size, n_ft_maps)
        pattern = np.repeat(np.expand_dims(patterns[:, l_idx], axis=0), np.shape(ft_maps)[0], axis=0)

        for i in range(pad_x, np.shape(ft_maps)[1] + pad_x):
            for j in range(pad_y, np.shape(ft_maps)[2] + pad_y):
                # get patch with matching size of the pattern
                x_pos = [i-pad_x, i+pad_x + 1]
                y_pos = [j-pad_y, j+pad_y + 1]
                patch = padded_ft_maps[:, x_pos[0]:x_pos[1], y_pos[0]:y_pos[1]]

                # expand patch and patterns to match dimensions (n_images, n_patterns, k_size, k_size, n_ft_maps)
                patch = np.repeat(np.expand_dims(patch, axis=1), n_patterns, axis=1)

                # compute diff between pattern and patch
                # (n_images, n_patterns, n_lmks, k_size, k_size, n_ft_map)
                diff = patch - pattern

                # compute activity
                flat_diff = np.reshape(diff, (np.shape(diff)[0], np.shape(diff)[1], -1))  # flatten 3 last dimensions to apply norm on# single axis
                activity = np.exp(-np.linalg.norm(flat_diff, ord=2, axis=2) ** 2 / 2 / sigma ** 2)

                # save activity
                activity_map[..., l, i - pad_x, j - pad_y] = activity

    return activity_map

