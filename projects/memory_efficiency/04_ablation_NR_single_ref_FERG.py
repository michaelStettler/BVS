# %%
import os
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.load_config import load_config
from utils.load_data import load_data
from utils.Metrics.accuracy import compute_accuracy
from utils.NormReference.reference_vectors import learn_ref_vector
from utils.NormReference.tuning_vectors import learn_tun_vectors
from utils.NormReference.tuning_vectors import optimize_tuning_vectors
from utils.NormReference.tuning_vectors import compute_projections


np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.memory_efficiency.04_ablation_NR_single_ref_FERG

Optimize the tuning direction over all dataset, using only a single ref
"""

avatar_type = None
avatar_type = 3
do_optimize = True


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
# select a ref vector (first from the avatar of interest)
ref_idx = np.arange(len(train_avatar))
ref_idx = ref_idx[train_avatar == avatar_type]
ref_vector = train_data[ref_idx[0]]
# duplicate ref_vector to match downstream code
ref_vectors = np.repeat(np.expand_dims(ref_vector, axis=0), 6, axis=0)
print("shape ref_vectors", np.shape(ref_vectors))


#%%
# learn tun vectors from one avatar
if avatar_type is not None:
    print("avatar name:", config['avatar_names'][avatar_type])
tun_vectors = learn_tun_vectors(train_data, train_label, ref_vectors, train_avatar, n_cat=n_cat, avatar_type_idx=avatar_type)
print("shape tun_vectors", np.shape(tun_vectors))

#%%
# compute projections
projections_preds = compute_projections(train_data, train_avatar, ref_vectors, tun_vectors,
                                        neutral_threshold=0,
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

#%% analysis
confusion_matrix(test_label, test_projections_preds)

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

#%%
if do_optimize:
    # optimize
    idx_array = np.zeros(n_cat).astype(int)
    best_idexes = []
    new_accuracy = 0
    for cat in range(1, 7):
        category_to_optimize = cat
        best_idx, accuracy, best_matching_idx = optimize_tuning_vectors(train_data, train_label, train_avatar,
                                                                        category_to_optimize, idx_array,
                                                                        len(config['avatar_names']),
                                                                        avatar_type_idx=avatar_type,
                                                                        ref_vectors=ref_vectors)
        print("category: {}, best_idx: {}, accuracy: {}".format(cat, best_idx, accuracy))
        print()
        idx_array[cat] = best_idx
        best_idexes.append(best_matching_idx)

        if accuracy > new_accuracy:
            new_accuracy = accuracy

    print("best_indexes".format(best_idexes))
    print("optimized from: {} to {}".format(train_accuracy, accuracy))

#%%
# set tuning vector with optimized direction
if avatar_type is None:
    idx_array = [0, ]  # NRE-I best
elif avatar_type == 0:
    idx_array = [0, 69, 351, 270, 784, 666, 0]  # NRE-Jules best
elif avatar_type == 1:
    idx_array = [0, 133, 583, 115, 747, 445, 218]  # NRE-malcolm best
elif avatar_type == 2:
    idx_array = [0, 369, 1138, 586, 293, 64, 745]  # NRE-ray best
elif avatar_type == 3:
    idx_array = [0, 292, 588, 686, 1324, 412, 967]  # NRE-aia best
elif avatar_type == 4:
    idx_array = [0, ]  # NRE-bonnie best
elif avatar_type == 5:
    idx_array = [0, 150, 519, 317, 365, 209, 272]  # NRE-mery best
# learn tun vectors from one avatar
opt_tun_vectors = learn_tun_vectors(train_data, train_label, ref_vectors, train_avatar,
                                    n_cat=n_cat,
                                    idx_array=idx_array,
                                    avatar_type_idx=avatar_type)
print("shape opt_tun_vectors", np.shape(opt_tun_vectors))

#%%
# compute optimized projections
opt_projections_preds = compute_projections(train_data, train_avatar, ref_vectors, opt_tun_vectors,
                                            neutral_threshold=5,
                                            verbose=False)
print("shape opt_projections_preds", np.shape(opt_projections_preds))

# compute accuracy
opt_train_accuracy = compute_accuracy(opt_projections_preds, train_label)
print("opt_train_accuracy accuracy", opt_train_accuracy)

#%%
# ------ test -----------
# compute test projections
opt_test_projections_preds = compute_projections(test_data, test_avatar, ref_vectors, opt_tun_vectors,
                                                 neutral_threshold=5,
                                                 verbose=False)
print("shape opt_test_projections_preds", np.shape(opt_test_projections_preds))

# compute accuracy
print("opt test accuracy", compute_accuracy(opt_test_projections_preds, test_label))

#%% analysis
confusion_matrix(test_label, opt_test_projections_preds)

#%%
# accuracy per avatar
for a in range(len(config['avatar_names'])):
    avatar_test_projections_preds = opt_test_projections_preds[test_avatar == a]
    avatar_test_label = test_label[test_avatar == a]
    accuracy = compute_accuracy(avatar_test_projections_preds, avatar_test_label)
    print("test accuracy avatar {}: {}".format(a, accuracy))

    conf_mat = confusion_matrix(avatar_test_label, avatar_test_projections_preds)
    print(conf_mat)
    print()
