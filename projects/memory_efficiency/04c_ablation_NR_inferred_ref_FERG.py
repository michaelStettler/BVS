# %%
import os
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.load_config import load_config
from utils.load_data import load_data
from utils.Metrics.accuracy import compute_accuracy
from utils.NormReference.reference_vectors import learn_ref_vector
from utils.NormReference.reference_vectors import infer_ref_vector
from utils.NormReference.reference_vectors import optimize_inferred_ref
from utils.NormReference.tuning_vectors import learn_tun_vectors
from utils.NormReference.tuning_vectors import compute_projections


np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.memory_efficiency.04c_ablation_NR_inferred_ref_FERG

Optimize the tuning direction over all dataset, using only a ref inferred from one expression
"""

avatar_type = None
avatar_type = 0
infer_expr = 1  # inferred expressions
#                   0 = neutral,
#                   1 = joy,
#                   2 = anger,
#                   3 = sadness
#                   4 = surprise
#                   5 = fear
#                   6 = disgust
do_optimize = True


#%%
# define configuration
config_file = 'NR_03_FERG_from_LMK_m0001.json'
#config_file = 'NR_03_FERG_from_LMK_w0001.json'
# load config
config = load_config(config_file, path='/Users/michaelstettler/PycharmProjects/BVS/BVS/configs/norm_reference')
#config = load_config(config_file, path='D:/PycharmProjects/BVS/configs/norm_reference')
print("-- Config loaded --")
print()

# declare variables
n_cat = 7

#%%
# Load data
train_data = load_data(config, get_raw=True, get_only_label=True)
train_label = train_data[1]
test_data = load_data(config, train=False, get_raw=True, get_only_label=True)
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
# for now it is ok to learn one for each identity, I can use them to check with the inferred one latter
ref_vectors = learn_ref_vector(train_data, train_label, train_avatar, len(config['avatar_names']))
print("shape ref_vectors", np.shape(ref_vectors))


#%%
# learn tun vectors from one avatar
if avatar_type is not None:
    print("avatar name:", config['avatar_names'][avatar_type])
tun_vectors = learn_tun_vectors(train_data, train_label, ref_vectors, train_avatar,
                                n_cat=n_cat,
                                avatar_type_idx=avatar_type)
print("shape tun_vectors", np.shape(tun_vectors))

#%%
# infer a ref vector from the tuning vectors
inferred_ref_vectors = infer_ref_vector(train_data, train_label, train_avatar, len(config['avatar_names']),
                                        avatar_of_int=avatar_type,
                                        expr_to_infer=infer_expr,
                                        tuning_vectors=tun_vectors)

print("ref_vectors[1]:")
print(ref_vectors[1])
print("inferred_ref_vectors[1]:")
print(inferred_ref_vectors[1])
print("difference")
print(ref_vectors[1] - inferred_ref_vectors[1])

ref_vectors = inferred_ref_vectors
print("shape ref_vectors", np.shape(ref_vectors))

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
# accuracy per avatar
for a in range(len(config['avatar_names'])):
    avatar_projections_preds = projections_preds[train_avatar == a]
    avatar_train_label = train_label[train_avatar == a]
    accuracy = compute_accuracy(avatar_projections_preds, avatar_train_label)
    print(f"train accuracy avatar {a}: {accuracy}")

    conf_mat = confusion_matrix(avatar_train_label, avatar_projections_preds)
    print(conf_mat)
    print()

#%%
# set tuning vector with optimized direction
if avatar_type is None:
    idx_array = [0, ]  # NRE-I best

# with accuracy computed on the full dataset
# elif avatar_type == 0:
#     idx_array = [0, 69, 351, 270, 784, 666, 0]  # NRE-Jules best
# elif avatar_type == 1:
#     idx_array = [0, 133, 583, 115, 747, 445, 218]  # NRE-malcolm best
# elif avatar_type == 2:
#     idx_array = [0, 369, 1138, 586, 293, 64, 745]  # NRE-ray best
# elif avatar_type == 3:
#     idx_array = [0, 292, 588, 686, 1324, 412, 967]  # NRE-aia best
# elif avatar_type == 4:
#     idx_array = [0, 490, 955, 1170, 1248, 813, 567]  # NRE-bonnie best
# elif avatar_type == 5:
#     idx_array = [0, 150, 519, 317, 365, 209, 272]  # NRE-mery best

# elif avatar_type == 0:
#     idx_array = [0, 913, 61, 57, 12, 0, 91]  # NRE-Jules best
# elif avatar_type == 1:
#     idx_array = [0, 133, 796, 5, 697, 56, 420]  # NRE-malcolm best
# elif avatar_type == 2:
#     idx_array = [0, 0, 3, 243, 648, 170, 630]  # NRE-ray best
# elif avatar_type == 3:
#     idx_array = [0, 32, 562, 227, 293, 865, 971]  # NRE-aia best
# elif avatar_type == 4:
#     idx_array = [0, 929, 1034, 299, 268, 439, 1471]  # NRE-bonnie best
# elif avatar_type == 5:
#     idx_array = [0, 210, 314, 581, 68, 375, 659]  # NRE-mery best


elif avatar_type == 0:
    idx_array = [0, 28, 21, 10, 0, 0, 0]  # NRE-Jules best
elif avatar_type == 1:
    idx_array = [0, 1, 148, 145, 244, 205, 749]  # NRE-malcolm best
elif avatar_type == 2:
    idx_array = [0, 36, 342, 40, 66, 0, 280]  # NRE-ray best
elif avatar_type == 3:
    idx_array = [0, 49, 853, 91, 731, 509, 260]  # NRE-aia best
elif avatar_type == 4:
    idx_array = [0, 903, 14, 627, 24, 427, 1214]  # NRE-bonnie best
elif avatar_type == 5:
    idx_array = [0, 210, 314, 581, 68, 375, 659]  # NRE-mery best

print(f"idx_array: {idx_array}")

# learn tun vectors from one avatar
opt_tun_vectors = learn_tun_vectors(train_data, train_label, ref_vectors, train_avatar,
                                    n_cat=n_cat,
                                    idx_array=idx_array,
                                    avatar_type_idx=avatar_type)
print("shape opt_tun_vectors", np.shape(opt_tun_vectors))

#%%
print("------ before inferred optimization ------")
# compute avatar optimized projections
opt_projections_preds = compute_projections(train_data, train_avatar, ref_vectors, opt_tun_vectors,
                                            neutral_threshold=5,
                                            verbose=False)
print("shape opt_projections_preds", np.shape(opt_projections_preds))

# compute accuracy
opt_train_accuracy = compute_accuracy(opt_projections_preds, train_label)
print("opt_train_accuracy accuracy", opt_train_accuracy)

# accuracy per avatar
for a in range(len(config['avatar_names'])):
    avatar_projections_preds = opt_projections_preds[train_avatar == a]
    avatar_train_label = train_label[train_avatar == a]
    accuracy = compute_accuracy(avatar_projections_preds, avatar_train_label)
    print(f"train accuracy avatar {a}: {accuracy}")

    conf_mat = confusion_matrix(avatar_train_label, avatar_projections_preds)
    print(conf_mat)
    print()

#%%
if do_optimize:
    inferred_ref = optimize_inferred_ref(train_data, train_label, train_avatar, len(config['avatar_names']),
                                         avatar_of_int=avatar_type,
                                         expr_to_infer=infer_expr,
                                         ref_vectors=ref_vectors,
                                         tun_vectors=tun_vectors)
    ref_vectors = inferred_ref

#%%
print("------ after inferred optimization ------")
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
