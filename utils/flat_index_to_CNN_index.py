def get_feature_map_index(index, n_feature_map, feature_map_size):
    """
    Retrieve positions of given index (i) of interest (n_i) from the flatten matrix (m, n)
    from its initial CNN shape (m, k, k, f)

    m := number of data points
    n := num_features = k * k * f
    k := kernel size
    f := num_feature maps

    :param index: the index of interest
    :param n_feature_map: the number of feature maps f
    :param feature_map_size: the size of the feature map (kernel size) k
    :return:
    """
    # get the position within a feature map
    # to unravel the flatten array, dividing by the number of features maps tells us how many
    # "repetitions" of feature map the index is. The remaining part of the division is the feature map index
    map_pos = int(index / n_feature_map)
    # get the feature map index
    # subtracting the index by the closes rounded number of feature maps gives us the feature map number
    f_index = index - map_pos * n_feature_map
    # get the position x and y within the feature map
    # by using the position of the feature map, we can divide by the size of the feature map to get the x pos and y pos
    x_pos = int(map_pos / feature_map_size)
    y_pos = map_pos - x_pos * feature_map_size

    return x_pos, y_pos, f_index