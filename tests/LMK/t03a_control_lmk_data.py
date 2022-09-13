import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.get_csv_file_FERG import edit_FERG_csv_file_from_config
from plots_utils.plot_BVS import display_image

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m tests.LMK.t03a_control_lmk_data
"""


def control_lmk_data(lmk_pos_data, images):
    if len(lmk_pos_data) != len(images):
        raise ValueError("length of images and data does not correspond! (data: {} vs. image: {})".format(
            len(lmk_pos_data), len(images)))

    im_ratio = 224/56
    for i in range(len(lmk_pos_data)):
        lmks_pos = lmk_pos_data[i] * im_ratio
        display_image(images[i], lmks=lmks_pos, pre_processing='VGG19')


if __name__ == '__main__':
    # declare variables
    avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
    avatar_name = avatar_names[2]
    n_images = 10

    # define configuration
    config_path = 'LMK_t03_create_lmk_data_m0001.json'
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # modify csv according to avatar name
    edit_FERG_csv_file_from_config(config, avatar_name)

    # define loading variables
    path = config['directory']

    # load data
    train_data = load_data(config)
    print("-- Data loaded --")
    print("len train_data[0]", len(train_data[0]))
    print()

    # load lmk dataset
    lmk_pos_data = np.load(os.path.join(path, 'saved_lmks_pos', avatar_name + "_lmk_pos.npy"))
    print("shape lmk_pos_data", np.shape(lmk_pos_data))

    control_lmk_data(lmk_pos_data[:n_images], train_data[0][:n_images])
