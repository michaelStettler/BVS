# %%
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from utils.load_config import load_config
from utils.load_data import load_data


np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.memory_efficiency.03_NR_from_LMK_on_FERG
"""


def compute_projections(inputs, avatars, ref_vectors, tun_vectors, nu=1, neutral_threshold=5, matching_t_vects=None,
                        verbose=False):
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

    proj_pred = np.argmax(projections, axis=1)

    # matching_t_vectors allows to add tuning vectors and match their predicted category to one of the other
    if matching_t_vects is not None:
        for match in matching_t_vects:
           proj_pred[proj_pred == match[0]] = match[1]

    return proj_pred


def compute_accuracy(preds, labels):
    n_correct = 0
    n_total = len(labels)

    for i in range(n_total):
        if preds[i] == labels[i]:
            n_correct += 1

    return n_correct/n_total


def add_matching_tun_vectors(tun_vectors, ref_vectors, tun_idx, ref_idx):
    for i in range(6):
        tun_vectors.append(train_data[tun_idx[i]] - ref_vectors[ref_idx])
        matching_t_vects.append([len(tun_vectors) - 1, i + 1])  # match column n with column (category) 1



#%%
# define configuration
config_file = 'NR_03_FERG_from_LMK_m0001.json'
# load config
config = load_config(config_file, path='/Users/michaelstettler/PycharmProjects/BVS/BVS/configs/norm_reference')
print("-- Config loaded --")
print()

# declare variables
n_cat = 7

#%%
# Load data
train_data = load_data(config, get_raw=True)
train_label = train_data[1]
test_data = load_data(config, train=False, get_raw=True)
test_label = test_data[1]
print("shape train_data[0]", np.shape(train_data[0]))
print("shape test_data[0]", np.shape(test_data[0]))

# load lmk pos
train_data = np.load(config['train_lmk_pos'])
test_data = np.load(config['test_lmk_pos'])
print("shape train_data", np.shape(train_data))
print("shape test_data", np.shape(test_data))

# load avatar types
train_avatar = np.load(config['train_avatar_type'])
test_avatar = np.load(config['test_avatar_type'])
print("shape train_avatar", np.shape(train_avatar))
print("shape test_avatar", np.shape(test_avatar))
print("-- Data loaded --")
print()

#%%
# learn neutral pattern
# ref_idx = np.reshape(np.argwhere(train_label == 0), -1)
# ref_vector = train_data[ref_idx[0]]
# print("ref vector", ref_vector)

print("shape train_data", np.shape(train_data))
ref_vectors = []
for a in range(len(config['avatar_names'])):
    # filter by avatar
    avatar_train_data = train_data[train_avatar == a]
    avatar_train_label = train_label[train_avatar == a]

    # filter by neutral ref
    avatar_ref_train_data = avatar_train_data[avatar_train_label == 0]

    # add only first from the avatar
    ref_vectors.append(avatar_ref_train_data[0])
print("shape ref_vectors", np.shape(ref_vectors))

#%%
# learn tun vectors from one avatar
avatar = 0
avatar_train_data = train_data[train_avatar == avatar]
avatar_train_label = train_label[train_avatar == avatar]

tun_images = []
tun_vectors = []
for i in range(n_cat):
    cat_idx = np.reshape(np.argwhere(avatar_train_label == i), -1)
    tun_vectors.append(avatar_train_data[cat_idx[0]] - ref_vectors[avatar])

matching_t_vects = None
matching_t_vects = []

# add Malcolm's set -> 96 12546 7829 1591 2674 34136
add_matching_tun_vectors(tun_vectors, ref_vectors, [96, 12546, 7829, 1591, 2674, 34136], 1)

# add Ray's set -> 14172 12476 37443 40043 35384 10239
add_matching_tun_vectors(tun_vectors, ref_vectors, [14172, 12476, 37443, 40043, 35384, 10239], 2)

# add Ray's set -> 11071 41271 27809 691 12415 36013
add_matching_tun_vectors(tun_vectors, ref_vectors, [11071, 41271, 27809, 691, 12415, 36013], 5)



tun_vectors = np.array(tun_vectors)
print("shape tun_vectors", np.shape(tun_vectors))
print("matching_t_vects", matching_t_vects)

#%%
# compute projections
projections_preds = compute_projections(train_data, train_avatar, ref_vectors, tun_vectors,
                                        neutral_threshold=5,
                                        matching_t_vects=matching_t_vects,
                                        verbose=False)
print("shape projections_preds", np.shape(projections_preds))

# compute accuracy
print("train accuracy", compute_accuracy(projections_preds, train_label))

#%%
# ------ test -----------
# compute test projections
test_projections_preds = compute_projections(test_data, test_avatar, ref_vectors, tun_vectors,
                                             neutral_threshold=5,
                                             matching_t_vects=matching_t_vects,
                                             verbose=False)
print("shape test_projections_preds", np.shape(test_projections_preds))

# compute accuracy
print("test accuracy", compute_accuracy(test_projections_preds, test_label))

#%% analysis
confusion_matrix(train_label, projections_preds)

#%%
# accuracy per avatar
for a in range(len(config['avatar_names'])):
    avatar_test_projections_preds = test_projections_preds[test_avatar == a]
    avatar_test_label = test_label[test_avatar == a]
    accuracy = compute_accuracy(avatar_test_projections_preds, avatar_test_label)
    print("test accuracy avatar {}: {}".format(a, accuracy))

    conf_mat = confusion_matrix(avatar_test_label, avatar_test_projections_preds)
    print(conf_mat)
    print()



