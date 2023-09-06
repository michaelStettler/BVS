'''
Some project-specific utility functions, mainly for paths and stuff
'''

import numpy as np

def get_computer_path(computer):
    if 'windows' in computer:
        computer_path = 'D:/Dataset/MorphingSpace'
        computer_letter = 'w'
    elif 'mac' in computer:
        computer_path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace'
        computer_letter = 'm'
    elif 'alex' in computer:
        computer_path = 'C:/Users/Alex/Documents/Uni/NRE/Dataset/MorphingSpace'
        computer_letter = 'a'
    return computer_path, computer_letter

def KL_divergence(p, q):
    log = np.log(p / q)
    log = np.nan_to_num(log) # replace nans by 0 bc the corresponding contribution to KL is 0
    return np.sum(p * log)


def compute_morph_space_KL_div(p, q):
    dim_x = np.shape(p)[0]
    dim_y = np.shape(p)[1]

    divergences = np.zeros((dim_x, dim_y))
    for x in range(dim_x):
        for y in range(dim_y):
            div = KL_divergence(p[x, y], q[x, y])
            divergences[x, y] = div

    return divergences