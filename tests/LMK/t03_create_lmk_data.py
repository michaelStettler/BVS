import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.get_csv_file_FERG import edit_FERG_csv_file_from_config
from utils.extraction_model import load_extraction_model
from utils.LMK.construct_LMK import create_lmk_dataset

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m tests.LMK.t03_create_lmk_data
"""

if __name__ == '__main__':
    # declare variables
    do_load = True
    im_ratio = 3
    k_size = (7, 7)
    lmk_type = 'FER'
    # define avatar
    avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
    avatar_name = avatar_names[4]

    # define configuration
    config_path = 'LMK_t03_create_lmk_data_m0001.json'
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # modify csv according to avatar name
    edit_FERG_csv_file_from_config(config, avatar_name)

    # define loading variables -> add all to config
    lmk_names = ['left_eyebrow_ext', 'left_eyebrow_int', 'right_eyebrow_int', 'right_eyebrow_ext',
                 'left_mouth', 'top_mouth', 'right_mouth', 'down_mouth',
                 'left_eyelid', 'right_eyelid']
    path = config['directory']

    # load data
    train_data = load_data(config)
    test_data = load_data(config, train=False)
    print("-- Data loaded --")
    print("len train_data[0]", len(train_data[0]))
    print("len test_data[0]", len(test_data[0]))
    print()

    # load feature extraction model
    v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
    v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
    size_ft = tuple(np.shape(v4_model.output)[1:3])
    print("-- Extraction Model loaded --")
    print("size_ft", size_ft)
    print()

    # load lmk parameters
    patterns = []
    sigma = []
    n_patterns = 0
    for lmk_name in lmk_names:
        print("lmk_name", lmk_name)
        pattern = np.load(os.path.join(path, 'saved_patterns', 'patterns_' + avatar_name + '_' + lmk_name + '.npy'))
        print("shape pattern", np.shape(pattern))
        patterns.append(pattern)
        sigma.append(
            int(np.load(os.path.join(path, 'saved_patterns', 'sigma_' + avatar_name + '_' + lmk_name + '.npy'))))

    print("-- Loaded optimized patterns finished --")
    print("len patterns", len(patterns))
    print("sigma", sigma)

    # create lmk dataset
    lmk_pos_data = create_lmk_dataset(train_data[0], v4_model, lmk_type, config, patterns, sigma)
    print("shape lmk_pos_data", np.shape(lmk_pos_data))
    np.save(os.path.join(path, 'saved_lmks_pos', avatar_name + "_lmk_pos"), lmk_pos_data)

    # create test lmk dataset
    lmk_pos_data = create_lmk_dataset(test_data[0], v4_model, lmk_type, config, patterns, sigma)
    print("shape test lmk_pos_data", np.shape(lmk_pos_data))
    np.save(os.path.join(path, 'saved_lmks_pos', "test_" + avatar_name + "_lmk_pos"), lmk_pos_data)

    # count number of images used
    n_total_pattern = 0
    patterns_count = np.zeros((len(avatar_names), len(lmk_names)))
    for a, a_name in enumerate(avatar_names):
        print("avatar name", a_name)
        # modify csv according to avatar name
        edit_FERG_csv_file_from_config(config, a_name)

        n_pattern = 0
        for l, lmk_name in enumerate(lmk_names):
            pattern = np.load(os.path.join(path, 'saved_patterns', 'patterns_' + a_name + '_' + lmk_name + '.npy'))
            patterns_count[a, l] = len(pattern)

            if len(pattern) > n_pattern:
                n_pattern = len(pattern)

        n_total_pattern += n_pattern
    print("n_total_pattern:", n_total_pattern)

    print("patterns_count")
    print(patterns_count)
    print("mean patterns_count:", np.mean(patterns_count))
    print("mean patterns_count (axis 0):", np.mean(patterns_count, axis=0))
    print("mean patterns_count: (axis 1)", np.mean(patterns_count, axis=1))
    n_count_per_landmark = np.sum(patterns_count, axis=0)
    print("average per landmark", np.mean(n_count_per_landmark), np.amin(n_count_per_landmark), np.amax(n_count_per_landmark))
