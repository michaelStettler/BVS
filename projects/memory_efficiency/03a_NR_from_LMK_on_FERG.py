# %%
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from utils.load_config import load_config
from utils.load_data import load_data


np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.memory_efficiency.03a_NR_from_LMK_on_FERG

Optimize the tuning direction over all dataset, no mather the avatar type
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


def learn_ref_vector(lmk_pos, labels, avatar_labels, n_avatar):
    ref_vectors = []

    for a in range(n_avatar):
        # filter by avatar
        avatar_train_data = lmk_pos[avatar_labels == a]
        avatar_train_label = labels[avatar_labels == a]

        # filter by neutral ref
        avatar_ref_train_data = avatar_train_data[avatar_train_label == 0]

        # add only first from the avatar
        ref_vectors.append(avatar_ref_train_data[0])

    return ref_vectors


def learn_tun_vectors(lmk_pos, labels, ref_vectors, avatar_labels, idx_array=None):
    # learn tun vectors
    tun_vectors = []

    img_idx = np.arange(len(lmk_pos))
    for i in range(n_cat):
        cat_idx = img_idx[labels == i]

        if idx_array is not None:
            avatar_type = avatar_labels[cat_idx[idx_array[i]]]
            tun_vectors.append(lmk_pos[cat_idx[idx_array[i]]] - ref_vectors[avatar_type])
        else:
            avatar_type = avatar_labels[cat_idx[0]]
            tun_vectors.append(lmk_pos[cat_idx[0]] - ref_vectors[avatar_type])

    return np.array(tun_vectors)


def optimize_tuning_vectors(lmk_pos, labels, avatar_labels, category_to_optimize, idx_array, n_cat):
    # learn neutral pattern
    ref_vector = learn_ref_vector(lmk_pos, labels, avatar_labels, n_cat)

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
        tun_vectors = learn_tun_vectors(lmk_pos, labels, ref_vector, avatar_labels, idx_array=idx_array)

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
# learn neutral patterns
ref_vectors = learn_ref_vector(train_data, train_label, train_avatar, len(config['avatar_names']))
print("shape ref_vectors", np.shape(ref_vectors))


#%%
# learn tun vectors from one avatar
tun_vectors = learn_tun_vectors(train_data, train_label, ref_vectors, train_avatar)
print("shape tun_vectors", np.shape(tun_vectors))

#%%
# compute projections
projections_preds = compute_projections(train_data, train_avatar, ref_vectors, tun_vectors,
                                        neutral_threshold=5,
                                        verbose=False)
print("shape projections_preds", np.shape(projections_preds))

# compute accuracy
train_accuracy = compute_accuracy(projections_preds, train_label)
print("train accuracy", train_accuracy)

#%%
# ------ test -----------
# compute test projections
test_projections_preds = compute_projections(test_data, test_avatar, ref_vectors, tun_vectors,
                                             neutral_threshold=5,
                                             verbose=False)
print("shape test_projections_preds", np.shape(test_projections_preds))

# compute accuracy
print("test accuracy", compute_accuracy(test_projections_preds, test_label))

#%%
# optimize
idx_array = np.zeros(n_cat).astype(int)
best_idexes = []
new_accuracy = 0
for cat in range(1, 7):
    category_to_optimize = cat
    best_idx, accuracy, best_matching_idx = optimize_tuning_vectors(train_data, train_label, train_avatar,
                                                                    category_to_optimize, idx_array,
                                                                    len(config['avatar_names']))
    print("category: {}, best_idx: {}, accuracy: {}".format(cat, best_idx, accuracy))
    print()
    idx_array[cat] = best_idx
    best_idexes.append(best_matching_idx)

    if accuracy > new_accuracy:
        new_accuracy = accuracy

print("best_indexes".format(best_idexes))
print("optimized from: {} to {}".format(train_accuracy, accuracy))

"""
    Aia (from 73.07 to 90.23): 
        - 3993  (30100)    
        - 6957  (42040)  
        - 418   (3043)   
        - 6214  (39900)  
        - 1945  (14713)  
        - 4406  (28686) 
"""

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



