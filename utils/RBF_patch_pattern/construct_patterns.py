import numpy as np


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


def construct_patterns(preds, pos, k_size, ratio=1):
    n_lmks = np.shape(pos)[1]
    n_patterns = np.shape(preds)[0]  # == n_images

    pad_x = int(k_size[0] / 2)
    pad_y = int(k_size[1] / 2)
    patterns = np.zeros((n_patterns, n_lmks, k_size[1], k_size[0], np.shape(preds)[-1]))

    for i in range(n_patterns):
        for j in range(n_lmks):
            x_pos = np.round(pos[i, j, 0] / ratio).astype(int)
            y_pos = np.round(pos[i, j, 1] / ratio).astype(int)
            x_patch = [x_pos - pad_x, x_pos + pad_x + 1]
            y_patch = [y_pos - pad_y, y_pos + pad_y + 1]
            patterns[i, j] = preds[i, y_patch[0]:y_patch[1], x_patch[0]:x_patch[1]]  # image x/y are shifted

    return patterns
