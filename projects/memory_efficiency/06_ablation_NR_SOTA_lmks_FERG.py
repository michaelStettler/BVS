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
run: python -m projects.memory_efficiency.06_ablation_NR_SOTA_lmks_FERG

Optimize the tuning direction over all dataset, using only part of the lmks
"""

# avatar_type = None
avatar_type = 5
do_optimize = True
conditions = ["FAN", "MediaPipe"]
cond = 1


#%%
# define configuration
config_file = 'NR_03_FERG_from_LMK_w0001.json'
# load config
config = load_config(config_file, path='D:/PycharmProjects/BVS/configs/norm_reference')
print("-- Config loaded --")
if avatar_type is not None:
    print(f"avatar_name: {config['avatar_names'][avatar_type]}, condition: {conditions[cond]}")
else:
    print(f"avatar_name: {avatar_type}, condition: {conditions[cond]}")
print()

if conditions[cond] == 'FAN':
    n_lmk = 34
elif conditions[cond] == 'MediaPipe':
    n_lmk = 36


config['train_lmk_pos'] = f"{config['directory']}/{conditions[cond]}_train_LMK_{n_lmk}.npy"
config['test_lmk_pos'] = f"{config['directory']}/{conditions[cond]}_test_LMK{n_lmk}.npy"

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
                                                                        avatar_type_idx=avatar_type)
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
if conditions[cond] == "FAN":
    if avatar_type is None:
        idx_array = [0, ]  # NRE-I best
    elif avatar_type == 0:
        idx_array = [0, ]  # NRE-Jules best
    elif avatar_type == 1:
        idx_array = [0, ]  # NRE-malcolm best
    elif avatar_type == 2:
        idx_array = [0, ]  # NRE-ray best
    elif avatar_type == 3:
        idx_array = [0, ]  # NRE-aia best
    elif avatar_type == 4:
        idx_array = [0, ]  # NRE-bonnie best
    elif avatar_type == 5:
        idx_array = [0, ]  # NRE-mery best

elif conditions[cond] == "MediPipe":
    if avatar_type is None:
        idx_array = [0, ]  # NRE-I best
    elif avatar_type == 0:
        idx_array = [0, ]  # NRE-Jules best
    elif avatar_type == 1:
        idx_array = [0, ]  # NRE-malcolm best
    elif avatar_type == 2:
        idx_array = [0, ]  # NRE-ray best
    elif avatar_type == 3:
        idx_array = [0, ]  # NRE-aia best
    elif avatar_type == 4:
        idx_array = [0, ]  # NRE-bonnie best
    elif avatar_type == 5:
        idx_array = [0, ]  # NRE-mery best

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
