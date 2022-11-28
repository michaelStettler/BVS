import numpy as np

"""
morphing space comes as follow: 

given -> map
0 HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0.0000.jpeg      (0, 0) -> (4, 4)
1 HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.25_Human_0.75.0000.jpg     (0, 1) -> (3, 4)
2 HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.5_Human_0.5.0000.jpg       (0, 2) -> (2, 4)
3 HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.75_Human_0.25.0000.jpg     (0, 3) -> (1, 4)
4 HumanAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0.0000.jpg       (0, 4) -> (0, 4)
5 HumanAvatar_Anger_0.25_Fear_0.75_Monkey_0.0_Human_1.0.0000.jpeg    (1, 0) -> (4, 3)
6 HumanAvatar_Anger_0.25_Fear_0.75_Monkey_0.25_Human_0.75.0000.jpeg  (1, 1) -> (3, 3)
7 HumanAvatar_Anger_0.25_Fear_0.75_Monkey_0.5_Human_0.5.0000.jpeg    (1, 2) -> (2, 3)
8 HumanAvatar_Anger_0.25_Fear_0.75_Monkey_0.75_Human_0.25.0000.jpeg  (1, 3)
9 HumanAvatar_Anger_0.25_Fear_0.75_Monkey_1.0_Human_0.0.0000.jpeg    (1, 4)
10 HumanAvatar_Anger_0.5_Fear_0.5_Monkey_0.0_Human_1.0.0000.jpeg     (2, 0)
11 HumanAvatar_Anger_0.5_Fear_0.5_Monkey_0.25_Human_0.75.0000.jpeg
12 HumanAvatar_Anger_0.5_Fear_0.5_Monkey_0.5_Human_0.5.0000.jpeg
13 HumanAvatar_Anger_0.5_Fear_0.5_Monkey_0.75_Human_0.25.0000.jpeg
14 HumanAvatar_Anger_0.5_Fear_0.5_Monkey_1.0_Human_0.0.0000.jpeg
15 HumanAvatar_Anger_0.75_Fear_0.25_Monkey_0.0_Human_1.0.0000.jpeg
16 HumanAvatar_Anger_0.75_Fear_0.25_Monkey_0.25_Human_0.75.0000.jpeg
17 HumanAvatar_Anger_0.75_Fear_0.25_Monkey_0.5_Human_0.5.0000.jpeg
18 HumanAvatar_Anger_0.75_Fear_0.25_Monkey_0.75_Human_0.25.0000.jpeg
19 HumanAvatar_Anger_0.75_Fear_0.25_Monkey_1.0_Human_0.0.0000.jpeg
20 HumanAvatar_Anger_1.0_Fear_0.0_Monkey_0.0_Human_1.0.0000.jpeg     (4, 0) -> (4, 0)
21 HumanAvatar_Anger_1.0_Fear_0.0_Monkey_0.25_Human_0.75.0000.jpg
22 HumanAvatar_Anger_1.0_Fear_0.0_Monkey_0.5_Human_0.5.0000.jpg
23 HumanAvatar_Anger_1.0_Fear_0.0_Monkey_0.75_Human_0.25.0000.jpg
24 HumanAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0.0000.jpg

i -> anger, from 0 to 1
j -> human, from 1 to 0  -> reversed but correct order from the plot
"""


def transform_morph_space_list2space(data):
    """
    helper function to transform the list received from the load data to the morphing space as displayed in the eLife
    paper

    :param data: list (25, ...)
    :return: space (5, 5, ...)
    """

    # control that the morph space can be transformed to a 5x5
    if len(data) != 25:
        raise ValueError("Value {} is not yet supported!".format(len(data)))

    # reshape the number of stimuli to a 5x5 grid
    data = np.reshape(data, (5, 5, np.shape(data)[1], np.shape(data)[2]))

    # transform space
    space = np.zeros(np.shape(data))
    for i in range(5):
        for j in range(5):
            # print("(i, j): ({}, {})".format(i,j))
            # print("space(i, j): ({}, {})".format(j, 4 - i))

            # transform space as displayed in the comment
            # space[i, j] = data[4 - i, j]
            space[4 - j, 4 - i] = data[i, j]

    return space


def get_morph_extremes_idx(condition=None):
    """
    idx: sorted (old)

    human avatar
    angry_human (HumanAvatar_Anger_1.0_Fear_0.0_Monkey_0.0_Human_1.0.0048): 48 (3048)
    fear_human (HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0.0062):  662 (62)
    angry_monkey (HumanAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0.0052): 3052 (3652)
    fear_monkey (HumanAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0.0051): 3651 (651)

    monkey avatar
    angry_human (MonkeyAvatar_Anger_1.0_Fear_0.0_Monkey_0.0_Human_1.0.0048): 3798 (6798)
    fear_human (MonkeyAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0.0048): 4398 (3798)
    angry_monkey (MonkeyAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0.0055): 6805 (7405)
    fear_monkey (MonkeyAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0.0037): 7387 (4387)

    """

    # return [3048, 62, 3652, 651, 6798, 3798, 7405, 4387]
    if condition is None:
        return [48, 662, 3052, 3651, 3798, 4398, 6805, 7387]
    elif condition == "human_orig":
        return [48, 662, 3052, 3651]
    elif condition == "monkey_orig":
        return [48, 648, 3055, 3637]
    elif condition == "human_equi":
        return [45, 640, 3056, 3648]
    elif condition == "monkey_equi":
        return [50, 650, 3056, 3656]
    else:
        raise NotImplementedError("condition {} is not valid".format(condition))


def get_NRE_from_morph_space(data, condition=None):
    """
    return only the extremes from each avatar and category
    
    """

    extremes_idx = get_morph_extremes_idx(condition=condition)
    filtered_data = data[0][extremes_idx]
    filtered_label = data[1][extremes_idx]

    return [filtered_data, filtered_label]

