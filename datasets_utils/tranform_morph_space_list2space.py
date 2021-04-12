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
