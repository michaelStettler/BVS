import numpy as np
from tqdm import tqdm

from utils.Metrics.accuracy import compute_accuracy
from utils.NormReference.reference_vectors import learn_ref_vector


def filter_by_avatar(lmk_pos, labels, avatar_labels, avatar_type_idx):
    # discard all image not from that avatar
    img_idx = np.arange(len(lmk_pos))
    avatar_idx = img_idx[avatar_labels == avatar_type_idx]

    # sort by avatar types
    lmk_pos = lmk_pos[avatar_idx]
    labels = labels[avatar_idx]
    avatar_labels = avatar_labels[avatar_idx]

    return lmk_pos, labels, avatar_labels


def learn_tun_vectors(lmk_pos, labels, ref_vectors, avatar_labels, n_cat=7, idx_array=None, avatar_type_idx=None):
    # learn tun vectors
    tun_vectors = []

    # discard all image not from that avatar
    if avatar_type_idx is not None:
        lmk_pos, labels, avatar_labels = filter_by_avatar(lmk_pos, labels, avatar_labels, avatar_type_idx)

    # learn tuning vector for each category
    img_idx = np.arange(len(lmk_pos))
    for i in range(n_cat):
        cat_idx = img_idx[labels == i]  # all image from this idx

        if idx_array is not None:
            # take the index from the idx_array
            avatar_type = avatar_labels[cat_idx[idx_array[i]]]
            tun_vectors.append(lmk_pos[cat_idx[idx_array[i]]] - ref_vectors[avatar_type])
        else:
            # take first image of all the dataset
            avatar_type = avatar_labels[cat_idx[0]]
            tun_vectors.append(lmk_pos[cat_idx[0]] - ref_vectors[avatar_type])

    return np.array(tun_vectors)


def optimize_tuning_vectors(lmk_pos, labels, avatar_labels, category_to_optimize, idx_array, n_cat, avatar_type_idx=None):
    # learn neutral pattern
    ref_vector = learn_ref_vector(lmk_pos, labels, avatar_labels, n_cat)

    print("len lmk_pos", len(lmk_pos))
    # discard all image not from that avatar
    if avatar_type_idx is not None:
        filt_lmk_pos, filt_labels, filt_avatar_labels = filter_by_avatar(lmk_pos, labels, avatar_labels, avatar_type_idx)
    else:
        filt_lmk_pos = lmk_pos
        filt_labels = labels
        filt_avatar_labels = avatar_labels
    print("len filt_lmk_pos", len(filt_lmk_pos))

    img_idx = np.arange(len(filt_labels))
    cat_img_idx = img_idx[filt_labels == category_to_optimize]
    print("len cat_img_idx", len(cat_img_idx))
    n_img_in_cat = len(filt_labels[filt_labels == category_to_optimize])
    print("n_img_in_cat", n_img_in_cat)

    accuracy = 0
    best_idx = 0
    for i in tqdm(range(n_img_in_cat)):
        # learn tun vectors
        idx_array[category_to_optimize] = i
        tun_vectors = learn_tun_vectors(filt_lmk_pos, filt_labels, ref_vector, filt_avatar_labels, idx_array=idx_array)

        # compute projections
        projections_preds = compute_projections(lmk_pos, avatar_labels, ref_vector, tun_vectors,
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


def compute_projections(inputs, avatars, ref_vectors, tun_vectors, nu=1, neutral_threshold=5, matching_t_vects=None,
                        verbose=False, return_proj_length=False):
    """

    :param inputs:
    :param avatars:
    :param ref_vectors:
    :param tun_vectors:
    :param nu:
    :param neutral_threshold:
    :param matching_t_vects: enable to add tuning vectors for same category
    :param verbose:
    :param return_proj_length:
    :return:
    """
    projections = []

    # normalize by norm of each landmarks
    norm_t = np.linalg.norm(tun_vectors, axis=2)

    # for each images
    for i in range(len(inputs)):
        inp = inputs[i]
        ref_vector = ref_vectors[avatars[i]]

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

    if return_proj_length:
        proj_pred = projections
    else:
        proj_pred = np.argmax(projections, axis=1)

        # matching_t_vectors allows to add tuning vectors and match their predicted category to one of the other
        if matching_t_vects is not None:
            for match in matching_t_vects:
               proj_pred[proj_pred == match[0]] = match[1]

    return proj_pred
