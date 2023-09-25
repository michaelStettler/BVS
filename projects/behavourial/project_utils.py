'''
Some project-specific utility functions, mainly for paths and stuff
'''

import numpy as np
from scipy.stats import wasserstein_distance

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

