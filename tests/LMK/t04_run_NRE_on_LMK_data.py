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
run: python -m tests.LMK.t04_run_NRE_on_LMK_data
"""


def compute_projections(input, ref_vector, tun_vectors, nu=1, neutral_threshold=5, verbose=False):
    projections = []

    # normalize by norm of each landmarks
    norm_t = np.linalg.norm(tun_vectors, axis=2)

    # for each images
    for i in range(len(input)):
        # compute relative vector (difference)
        diff = input[i] - ref_vector
        proj = []
        # for each category
        for j in range(len(tun_vectors)):
            proj_length = 0
            # for each landmarks
            for k in range(len(ref_vector)):
                if norm_t[j, k] != 0.0:
                    f = np.dot(diff[k], tun_vectors[j, k]) / norm_t[j, k]
                    f = np.power(f, nu)
                else:
                    f = 0
                proj_length += f
            # ReLu activation
            if proj_length < 0:
                proj_length = 0
            proj.append(proj_length)
        projections.append(proj)

    projections = np.array(projections)
    # apply neutral threshold
    projections[projections < neutral_threshold] = 0

    if verbose:
        print("projections", np.shape(projections))
        print(projections)
        print()
        print("max_projections", np.shape(np.amax(projections, axis=1)))
        print(np.amax(projections, axis=1))

    proj_pred = np.argmax(projections, axis=1)

    return proj_pred


def compute_accuracy(preds, labels):
    n_correct = 0
    n_total = len(labels)

    for i in range(n_total):
        if preds[i] == labels[i]:
            n_correct += 1

    return n_correct/n_total


if __name__ == '__main__':
    # declare variables
    avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
    avatar_name = avatar_names[2]
    n_cat = 7

    # define configuration
    config_path = 'LMK_t04_run_NRE_on_LMK_data_m0001.json'
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # modify csv according to avatar name
    edit_FERG_csv_file_from_config(config, avatar_name)

    # load data
    train_data = load_data(config)
    test_data = load_data(config, train=False)
    print("-- Data loaded --")
    print("len train_data[0]", len(train_data[0]))
    print("len test_data[0]", len(test_data[0]))
    print()


    # load lmk dataset
    lmk_pos_data = np.load(os.path.join(config['directory'], config['LMK_data_directory'], avatar_name + "_lmk_pos.npy"))
    test_lmk_pos_data = np.load(os.path.join(config['directory'], config['LMK_data_directory'], "test_" + avatar_name + "_lmk_pos.npy"))
    print("shape lmk_pos_data", np.shape(lmk_pos_data))
    print("shape test_lmk_pos_data", np.shape(test_lmk_pos_data))

    # learn neutral pattern
    ref_idx = np.reshape(np.argwhere(train_data[1] == 0), -1)
    ref_vector = lmk_pos_data[ref_idx[0]]

    # learn tun vectors
    tun_vectors = []
    for i in range(n_cat):
        cat_idx = np.reshape(np.argwhere(train_data[1] == i), -1)
        print("category ({}) idx: {}".format(i, cat_idx[0]))
        tun_vectors.append(lmk_pos_data[cat_idx[0]] - ref_vector)
    tun_vectors = np.array(tun_vectors)
    print("shape tun_vectors", np.shape(tun_vectors))

    # compute projections
    projections_preds = compute_projections(lmk_pos_data, ref_vector, tun_vectors,
                                            neutral_threshold=5,
                                            verbose=False)
    print("shape projections_preds", np.shape(projections_preds))

    # compute accuracy
    print("train accuracy", compute_accuracy(projections_preds, train_data[1]))

    # ------ test -----------
    # compute test projections
    test_projections_preds = compute_projections(test_lmk_pos_data, ref_vector, tun_vectors,
                                                 neutral_threshold=5,
                                                 verbose=False)
    print("shape test_projections_preds", np.shape(test_projections_preds))

    # compute accuracy
    print("test accuracy", compute_accuracy(test_projections_preds, test_data[1]))


    n_test = 10
    test_projections_preds = compute_projections(test_lmk_pos_data[:n_test], ref_vector, tun_vectors,
                                                 neutral_threshold=5,
                                                 verbose=True)
    print(test_data[1][:n_test])


