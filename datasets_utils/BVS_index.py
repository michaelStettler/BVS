def get_FER_ref_pos(fer_positions, type, scale):
    """
    helper function to retrieve the reference structure from the dataset

    :param fer_positions:
    :param fer_ref:
    :param scale:
    :return:
    """
    scale_factor = 5
    expression_factor = 7  # (num_expression)
    factor = type * expression_factor * scale_factor

    if scale == 0.8:
        factor += 1 * expression_factor
    elif scale == 0.9:
        factor += 2 * expression_factor
    elif scale == 1.0:
        factor += 0 * expression_factor
    elif scale == 1.1:
        factor += 3 * expression_factor
    elif scale == 1.2:
        factor += 4 * expression_factor

    return fer_positions[factor]