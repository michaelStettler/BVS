"""
2021/02/18
This function calculates the position within a feature map.
Used in t08_calculate_position_demo.py and compute_cnn_feature_map_response.py

Note: To allow for different shapes of response, expand dimension
"""

import numpy as np


def calculate_position(response, mode="weighted average", return_mode="array"):
    """
    This function calculates the position within a feature map. Different modes are available.
    :param response: response vector of v4, shape (n_images, x, y, n_feature_maps)
    :param mode: how to compute position, options:
        "maximum"           --> position of maximal value
        "maximum{n_max}"    --> average position of n_max highest values
        "weighted average"  --> average position weighted by activation values
    :param return_mode:
        "flattened" --> return flattened position
        "xy"        --> return coordinates
        "xy float"  --> return coordinates as float
        "xy float flat" -> return coordinates as float flattened
        "array"     --> return  zero-array with 1 at coordinates
    :return:
    """
    # ------------------------------------------------------------------------------------------------------------------
    # calculate position into position_flattened and index_mean_x_float###
    # TODO: To allow for other shapes of response, expand dimension
    # save original shape
    original_shape = response.shape
    # flatten dimensions that encode x and y of each image and each feature map
    response = np.reshape(response, (response.shape[0], -1, response.shape[3]))
    if 'maximum' in mode:
        if 'maximum' == mode:
            n_max = 1
        else:
            n_max = int(mode[7:])
        if n_max == 1:
            # compute maximum into flattened dimension
            position_flattened = np.argmax(response, axis=1)
        else:
            # sort by highest response to select highest values
            index_flat_max = np.argsort(response, axis=1)
            # select n_max values
            index_flat_max = index_flat_max[:, -n_max:, :]
            # unravel indices
            index_ravel_max = np.unravel_index(index_flat_max, original_shape[1:3])
            # compute average according to coordinates
            index_mean_x_float = np.mean(index_ravel_max[0], axis=1)
            index_mean_y_float = np.mean(index_ravel_max[1], axis=1)
            # round positions
            index_mean_x = np.rint(index_mean_x_float).astype(int)
            index_mean_y = np.rint(index_mean_y_float).astype(int)
            # flatten array
            position_flattened = np.ravel_multi_index((index_mean_x, index_mean_y), original_shape[1:3])

    elif mode == "weighted average":
        response += 1e-7  # just to avoid to divide by 0 if the feature map is full of zeros
        # initialize indices (x, y) for each entry
        indices = np.unravel_index(np.arange(response.shape[1]), original_shape[1:3])  # (2, n**2), with n the ft size
        # compute weighted average of the indices, weighted by the neuron activation
        index_mean_x_float = np.average(response, axis=1, weights=indices[0]) * np.sum(indices[0]) / np.sum(response, axis=1)
        index_mean_y_float = np.average(response, axis=1, weights=indices[1]) * np.sum(indices[1]) / np.sum(response, axis=1)
        # round positions
        index_mean_x = np.rint(index_mean_x_float).astype(int)
        index_mean_y = np.rint(index_mean_y_float).astype(int)
        # flatten array
        position_flattened = np.ravel_multi_index((index_mean_x, index_mean_y), original_shape[1:3])

    else:
        raise ValueError(f'mode={mode} is no valid value')

    # ------------------------------------------------------------------------------------------------------------------
    # return property specified in return_mode
    if return_mode == "flattened":
        # ravel the indices such that they can be applied to flattened array
        return position_flattened
    elif return_mode == "xy":
        return np.unravel_index(position_flattened, original_shape[1:3])
    elif return_mode == "xy float":
        try:
            # return np.array((index_mean_x_float[:, 0], index_mean_y_float[:, 0])).transpose()  # @Tim: return only positions from first feature map?
            stacked = np.stack((index_mean_x_float, index_mean_y_float))
            return np.swapaxes(stacked, 0, 1)
        except UnboundLocalError:
            return np.unravel_index(position_flattened, original_shape[1:3])
    elif return_mode == "xy float flat":
        stacked = np.stack((index_mean_x_float, index_mean_y_float))
        stacked = np.swapaxes(stacked, 0, 1)
        return np.reshape(stacked, (len(stacked), -1))
    elif return_mode == "array":
        # init plot vector
        position_array = np.zeros(response.shape)
        # set these indices to 1 in the plot vector
        np.put_along_axis(position_array, np.expand_dims(position_flattened, axis=1), 1, axis=1)
        #    np.put_along_axis(vector_plot, np.expand_dims(index_flat_max, axis=1), 1, axis=1)
        # reshape to non-flattened shape
        position_array = position_array.reshape(original_shape)
        return position_array
    else:
        raise ValueError(f'mode={return_mode} is not a valid output')
