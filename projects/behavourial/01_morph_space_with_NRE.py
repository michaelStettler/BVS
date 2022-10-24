import os
import numpy as np
import tensorflow as tf

from datasets_utils.morphing_space import get_morph_extremes_idx
from datasets_utils.morphing_space import get_NRE_from_morph_space

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.RBF_patch_pattern.load_RBF_patterns import load_RBF_patterns_and_sigma
from utils.RBF_patch_pattern.construct_patterns import create_RBF_LMK
from utils.LMK.construct_LMK import create_lmk_dataset
from utils.LMK.construct_LMK import get_identity_and_pos
from utils.NormReference.reference_vectors import learn_ref_vector
from utils.NormReference.tuning_vectors import learn_tun_vectors
from utils.NormReference.tuning_vectors import compute_projections

from plots_utils.plot_BVS import display_images

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.01_morph_space_with_NRE
"""

#%% declare script variables
show_plot = False
load_RBF_pattern = True
train_RBF_pattern = False
save_RBF_pattern = False
load_FR_pathway = True
save_FR_pos = False
load_FER_pos = True
save_FER_pos = False

#%% declare hyper parameters
n_iter = 2

#%% import config
config_path = 'BH_01_morph_space_with_NRE_m0001.json'
# load config
config = load_config(config_path, path='configs/behavourial')
print("-- Config loaded --")
print()

#%% import data
train_data = load_data(config)
print("-- Data loaded --")
print("len train_data[0]", len(train_data[0]))
print()

#%% split training for LMK and norm base
NRE_train = get_NRE_from_morph_space(train_data)
LMK_train = train_data  # take all

print("-- Data Split --")
print("len NRE_train[0]", len(NRE_train[0]))
print("NRE_train[1]")
print(NRE_train[1])
print()

# #%% display NRE training images
# if show_plot:
#     display_images(NRE_train[0], pre_processing='VGG19', n_max_col=4)

#%% load feature extraction model
v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(v4_model.output)[1:3])
print("-- Extraction Model loaded --")
print("size_ft", size_ft)
print()


#%% get RBF LMK detector
FR_patterns_list, FR_sigma_list, FER_patterns_list, FER_sigma_list = None, None, None, None
if load_RBF_pattern:
    print("load LMKs")
    FR_patterns_list, FR_sigma_list, FER_patterns_list, FER_sigma_list = \
        load_RBF_patterns_and_sigma(config, avatar_name=["human", "monkey"])

if train_RBF_pattern:
    print("create patterns")
    FR_patterns_list, FR_sigma_list, FER_patterns_list, FER_sigma_list = \
        create_RBF_LMK(config, LMK_train, v4_model,
                       n_iter=n_iter,
                       FR_patterns=FR_patterns_list,
                       FR_sigma=FR_sigma_list,
                       FER_patterns=FER_patterns_list,
                       FER_sigma=FER_sigma_list,
                       save=save_RBF_pattern)

print("len FR_patterns_list", len(FR_patterns_list))
print("len FR_patterns_list[0]", len(FR_patterns_list[0]))
print("len FR_patterns_list[1]", len(FR_patterns_list[1]))
print("len FER_patterns_list", len(FER_patterns_list))
print("len FER_sigma_list", len(FER_sigma_list))
print()

#%% get identity and positions from the FR Pathway
extremes_idx = get_morph_extremes_idx()
if load_FR_pathway:
    FR_pos = np.load(os.path.join(config["directory"], config["LMK_data_directory"], "FR_LMK_pos.npy"))
    face_ids = np.load(os.path.join(config["directory"], config["LMK_data_directory"], "face_identities.npy"))
    face_positions = np.load(os.path.join(config["directory"], config["LMK_data_directory"], "face_positions.npy"))
else:
    FR_pos, face_ids, face_positions = get_identity_and_pos(train_data[0], v4_model, config, FR_patterns_list, FR_sigma_list)

    if save_FR_pos:
        np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FR_LMK_pos"), FR_pos)
        np.save(os.path.join(config["directory"], config["LMK_data_directory"], "face_positions"), face_positions)
        np.save(os.path.join(config["directory"], config["LMK_data_directory"], "face_identities"), face_ids)
print("shape FR_pos", np.shape(FR_pos))
print("shape face_identities", np.shape(face_ids))
print("shape face_positions", np.shape(face_positions))
print()

#%% predict LMK pos
if load_FER_pos:
    FER_pos = np.load(os.path.join(config["directory"], config["LMK_data_directory"], "FER_LMK_pos.npy"))
else:
    FER_pos = create_lmk_dataset(train_data[0], v4_model, "FER", config, FER_patterns_list, FER_sigma_list)

    if save_FER_pos:
        np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FER_LMK_pos"), FER_pos)
print("shape FER_pos", np.shape(FER_pos))
print()

#%% learn reference vector
ref_idx = [0, 3750]
avatar_labels = np.array([0, 1]).astype(int)
ref_vectors = learn_ref_vector(FER_pos[ref_idx], train_data[1][ref_idx], avatar_labels=avatar_labels, n_avatar=2)
print("shape ref_vectors", np.shape(ref_vectors))

#%% plot landmarks on NRE_train
if show_plot:
    extremes_idx = get_morph_extremes_idx()
    NRE_train_img = train_data[0][extremes_idx]
    NRE_lmk_pos = FER_pos[extremes_idx] * 224 / 56
    NRE_ref_pos = ref_vectors * 224 / 56
    NRE_ref_pos = np.repeat(NRE_ref_pos, 4, axis=0)  # expand ref_pos for each images

    display_images(NRE_train_img,
                   lmks=NRE_lmk_pos,
                   ref_lmks=NRE_ref_pos,
                   n_max_col=4,
                   pre_processing="VGG19")

#%% learn tuning vectors
tun_idx = [0] + get_morph_extremes_idx()
tun_vectors = learn_tun_vectors(FER_pos[tun_idx], train_data[1][tun_idx], ref_vectors, face_ids[tun_idx], n_cat=5)
print("shape tun_vectors", np.shape(tun_vectors))

#%% Compute projections
# todo remove face positions from the FER_pos
NRE_proj = compute_projections(FER_pos, face_ids, ref_vectors, tun_vectors, return_proj_length=True)
print("shape NRE_proj", np.shape(NRE_proj))



