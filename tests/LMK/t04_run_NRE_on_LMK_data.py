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


def compute_projections(inputs, ref_vector, tun_vectors, nu=1, neutral_threshold=5, verbose=False):
    projections = []

    # normalize by norm of each landmarks
    norm_t = np.linalg.norm(tun_vectors, axis=2)

    # for each images
    for i in range(len(inputs)):
        inp = inputs[i]

        # replace occluded lmk by ref value
        inp[inp < 0] = ref_vector[inp < 0]

        # compute relative vector (difference)
        diff = inp - ref_vector
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


def learn_ref_vector(lmk_pos, labels):
    # learn neutral pattern
    ref_idx = np.reshape(np.argwhere(labels == 0), -1)
    ref_vector = lmk_pos[ref_idx[0]]

    return ref_vector


def learn_tun_vectors(lmk_pos, labels, ref_vector, idx_array=None):
    # learn tun vectors
    tun_vectors = []
    for i in range(n_cat):
        cat_idx = np.reshape(np.argwhere(labels == i), -1)

        if idx_array is not None:
            tun_vectors.append(lmk_pos[cat_idx[idx_array[i]]] - ref_vector)
        else:
            tun_vectors.append(lmk_pos[cat_idx[0]] - ref_vector)

    return np.array(tun_vectors)


def optimize_tuning_vectors(lmk_pos, labels, category_to_optimize, idx_array):
    # learn neutral pattern
    ref_vector = learn_ref_vector(lmk_pos, labels)

    img_idx = np.arange(len(labels))
    cat_img_idx = img_idx[labels == category_to_optimize]
    print("len cat_img_idx", len(cat_img_idx))
    n_img_in_cat = len(labels[labels == category_to_optimize])
    print("n_img_in_cat", n_img_in_cat)

    accuracy = 0
    best_idx = 0
    for i in tqdm(range(n_img_in_cat)):
        # learn tun vectors
        idx_array[category_to_optimize] = i
        tun_vectors = learn_tun_vectors(lmk_pos, labels, ref_vector, idx_array=idx_array)

        # compute projections
        projections_preds = compute_projections(lmk_pos, ref_vector, tun_vectors,
                                                neutral_threshold=5,
                                                verbose=False)
        # compute accuracy
        new_accuracy = compute_accuracy(projections_preds, labels)

        if new_accuracy > accuracy:
            print("new accuracy: {}, idx: {} (matching {})".format(new_accuracy, i, cat_img_idx[i]))
            accuracy = new_accuracy
            best_idx = i

    print("best idx:", best_idx)
    print("best accuracy:", accuracy)
    return best_idx, accuracy, cat_img_idx[best_idx]


if __name__ == '__main__':
    # declare variables
    avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
    avatar_name = avatar_names[5]
    print("avatar_name:", avatar_name)
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
    ref_vector = learn_ref_vector(lmk_pos_data, train_data[1])

    # learn tun vectors
    tun_vectors = learn_tun_vectors(lmk_pos_data, train_data[1], ref_vector)
    print("shape tun_vectors", np.shape(tun_vectors))

    # compute projections
    projections_preds = compute_projections(lmk_pos_data, ref_vector, tun_vectors,
                                            neutral_threshold=5,
                                            verbose=False)
    print("shape projections_preds", np.shape(projections_preds))

    # compute accuracy
    train_accuracy = compute_accuracy(projections_preds, train_data[1])
    print("train accuracy:", train_accuracy)

    # ------ test -----------
    # compute test projections
    test_projections_preds = compute_projections(test_lmk_pos_data, ref_vector, tun_vectors,
                                                 neutral_threshold=5,
                                                 verbose=False)
    print("shape test_projections_preds", np.shape(test_projections_preds))

    # compute accuracy
    print("test accuracy", compute_accuracy(test_projections_preds, test_data[1]))

    # test optimization of selected tuning vectors
    idx_array = np.zeros(n_cat).astype(int)
    best_idexes = []
    new_accuracy = 0
    for cat in range(1, 7):
        category_to_optimize = cat
        best_idx, accuracy,  best_matching_idx = optimize_tuning_vectors(lmk_pos_data, train_data[1], category_to_optimize, idx_array)
        print("category: {}, best_idx: {}, accuracy: {}".format(cat, best_idx, accuracy))
        print()
        idx_array[cat] = best_idx
        best_idexes.append(best_matching_idx)

        if accuracy > new_accuracy:
            new_accuracy = accuracy

    print("best_idexes", best_idexes)
    np.save(os.path.join(config['directory'], 'best_indexes', avatar_name + '_best_idx'), best_idexes)


    print("optimization increased from {} to {}".format(train_accuracy, new_accuracy))

