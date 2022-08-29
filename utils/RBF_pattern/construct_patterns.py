import numpy as np
from tqdm import tqdm


def construct_pattern(pred, pos, size, ratio=1):
    # get padding
    pad_x = int(size[0] / 2)
    pad_y = int(size[1] / 2)

    # get lmk pos relative to the latent space
    x_pos = np.round(pos[0] / ratio).astype(int)
    y_pos = np.round(pos[1] / ratio).astype(int)

    # construct patch
    x_patch = [x_pos - pad_x, x_pos + pad_x + 1]
    y_patch = [y_pos - pad_y, y_pos + pad_y + 1]

    # return pattern
    return pred[0, y_patch[0]:y_patch[1], x_patch[0]:x_patch[1]]  # image x/y are shifted


def construct_patterns(preds, pos, size, ratio=1):
    n_lmks = np.shape(pos)[1]
    n_patterns = np.shape(preds)[0]  # == n_images

    pad_x = int(size[0] / 2)
    pad_y = int(size[1] / 2)
    patterns = np.zeros((n_patterns, n_lmks, size[1], size[0], np.shape(preds)[-1]))

    for i in range(n_patterns):
        for j in range(n_lmks):
            x_pos = np.round(pos[i, j, 0] / ratio).astype(int)
            y_pos = np.round(pos[i, j, 1] / ratio).astype(int)
            x_patch = [x_pos - pad_x, x_pos + pad_x + 1]
            y_patch = [y_pos - pad_y, y_pos + pad_y + 1]
            patterns[i, j] = preds[i, y_patch[0]:y_patch[1], x_patch[0]:x_patch[1]]  # image x/y are shifted

    return patterns


def compute_RBF_pattern_activity_maps(ft_maps, patterns, sigmas=1, lmk_idx=None, disable_tqdm=False):
    """
    The function "convolve" with a RBF kernel all the patterns within the given feature maps

    small sigma => more selective
    large sigma => more robust

    :param ft_maps: (n_images, size, size, n_ft_maps)
    :param patterns: (n_patterns, n_lmks, k_size, k_size, n_ft_maps)
    :param sigmas:
    :return: activity_map (n_images, n_patterns, n_lmks, size, size)
    """
    # declare activity map
    if lmk_idx is None:
        activity_map = np.zeros((np.shape(ft_maps)[0], np.shape(patterns)[0], np.shape(patterns)[1], np.shape(ft_maps)[1], np.shape(ft_maps)[2]))
    else:
        activity_map = np.zeros((np.shape(ft_maps)[0], np.shape(patterns)[0], 1, np.shape(ft_maps)[1], np.shape(ft_maps)[2]))

    # compute padding
    pad_x = int(np.shape(patterns)[2] / 2)
    pad_y = int(np.shape(patterns)[3] / 2)

    # pad feature maps
    padded_ft_maps = np.zeros((np.shape(ft_maps)[0], np.shape(ft_maps)[1] + 2 * pad_x, np.shape(ft_maps)[2] + 2 * pad_y, np.shape(ft_maps)[3]))

    # compensate for padding of zeros
    if pad_x == 0 and pad_y== 0:
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

        if isinstance(sigmas,int):
            sigma = sigmas
        elif isinstance(sigmas, float):
            sigma = sigmas
        elif isinstance(sigmas, np.int64):
            sigma = sigmas
        elif isinstance(sigmas, list):
            sigma = sigmas[l]
        elif isinstance(sigmas, np.ndarray):
            sigma = sigmas[l]
        elif isinstance(sigmas,tuple):
            sigma = sigmas[l]
        else:
            raise NotImplementedError("sigmas of type {} is not implemented yet".format(type(sigmas)))

        # expand pattern to match dimension to patch
        pattern = np.repeat(np.expand_dims(patterns[:, l_idx], axis=0), np.shape(ft_maps)[0], axis=0)

        # compute for each position of the feature maps
        for i in range(pad_x, np.shape(ft_maps)[1] + pad_x):
            for j in range(pad_y, np.shape(ft_maps)[2] + pad_y):
                # get patch with matching size of the pattern
                x_pos = [i-pad_x, i+pad_x + 1]
                y_pos = [j-pad_y, j+pad_y + 1]
                patch = padded_ft_maps[:, x_pos[0]:x_pos[1], y_pos[0]:y_pos[1]]

                # expand patch and patterns to match dimensions (n_images, n_patterns, k_size, k_size, n_ft_maps)
                patch = np.repeat(np.expand_dims(patch, axis=1), len(patterns), axis=1)

                # compute diff between pattern and patch
                # (n_images, n_patterns, n_lmks, size, size, n_ft_map)
                diff = patch - pattern

                # compute activity
                flat_diff = np.reshape(diff, (np.shape(diff)[0], np.shape(diff)[1], -1))  # flatten 3 last dimensions to apply norm on# single axis
                activity = np.exp(-np.linalg.norm(flat_diff, ord=2, axis=2) ** 2 / 2 / sigma ** 2)

                # save activity
                activity_map[..., l, i - pad_x, j - pad_y] = activity

    return activity_map