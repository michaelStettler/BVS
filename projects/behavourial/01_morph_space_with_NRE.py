import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
show_plot = True
load_RBF_pattern = True
train_RBF_pattern = True
save_RBF_pattern = True
load_FR_pathway = False
save_FR_pos = True
load_FER_pos = False
save_FER_pos = False

#%% declare hyper parameters
n_iter = 2
max_sigma = None
max_sigma = 3000

#%% import config
config_path = 'BH_01_morph_space_with_NRE_m0001.json'
# load config
config = load_config(config_path, path='configs/behavourial')
print("-- Config loaded --")
print()

config["FR_lmk_name"] = ["left_eye", "right_eye", "nose"]
config["FR_lmk_name"] = []

config["FER_lmk_name"] = ["left_eyebrow_ext", "left_eyebrow_int", "right_eyebrow_int", "right_eyebrow_ext",
                 "left_mouth", "top_mouth", "right_mouth", "down_mouth",
                 "left_eyelid", "right_eyelid"]
config["FER_lmk_name"] = ["left_mouth", "right_mouth", "left_eyelid", "right_eyelid"]
# config["FER_lmk_name"] = ["right_eyebrow_int", "right_eyebrow_ext",
#                  "left_mouth", "top_mouth", "right_mouth", "down_mouth",
#                  "left_eyelid", "right_eyelid"]

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
                       max_sigma=max_sigma,
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
    extremes_idx = [0] + extremes_idx[:4] + [3750] + extremes_idx[4:]
    print("extremes_idx", extremes_idx)
    NRE_train_img = train_data[0][extremes_idx]
    NRE_lmk_pos = FER_pos[extremes_idx] * 224 / 56
    NRE_ref_pos = ref_vectors * 224 / 56
    NRE_ref_pos = np.repeat(NRE_ref_pos, 5, axis=0)  # expand ref_pos for each images

    display_images(NRE_train_img,
                   lmks=NRE_lmk_pos,
                   ref_lmks=NRE_ref_pos,
                   n_max_col=5,
                   pre_processing="VGG19")

#%% learn tuning vectors
tun_idx = [0] + get_morph_extremes_idx()[:4]
tun_vectors = learn_tun_vectors(FER_pos[tun_idx], train_data[1][tun_idx], ref_vectors, face_ids[tun_idx], n_cat=5)
print("shape tun_vectors", np.shape(tun_vectors))
print(tun_vectors)

#%% Compute projections
# todo remove face positions from the FER_pos
NRE_proj = compute_projections(FER_pos, face_ids, ref_vectors, tun_vectors, return_proj_length=True)
print("shape NRE_proj", np.shape(NRE_proj))

#%%
# plot test human fear
plt.figure()
plt.plot(NRE_proj[:150])
plt.legend(["HA", "HF", "MA", "MF"])
plt.show()

#%%
from plots_utils.plot_tuning_signatures import plot_tuning_signatures
import matplotlib.cm as cm
from plots_utils.plot_ft_map_pos import _set_fig_name
from plots_utils.plot_ft_map_pos import _set_save_folder


def plot_tuning_signatures(data, ref_tuning=None, fig_name=None, save_folder=None, fig_size=(5, 5), dpi=600):
    print("shape data", np.shape(data))

    # set images name
    images_name = _set_fig_name(fig_name, 'signature_tuning.png')

    # set save folder
    save_folder = _set_save_folder(save_folder, '')

    # create colors
    colors = cm.rainbow(np.linspace(0, 1, len(data)))

    # create figure
    plt.figure(figsize=fig_size, dpi=dpi)

    for i, d, c in zip(np.arange(len(data)), data, colors):
        plt.scatter(d[1], -d[0], color=c)
        plt.arrow(0, 0, d[1], -d[0], color=c, linewidth=1)
        plt.text(d[1] * 1.1, -d[0] * 1.1, str(i), color=c)

        if ref_tuning is not None:
            plt.scatter(ref_tuning[i, 1], -ref_tuning[i, 0], color=c)
            plt.arrow(0, 0, ref_tuning[i, 1], -ref_tuning[i, 0], color=c, linewidth=1, linestyle=':')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.savefig(os.path.join(save_folder, images_name))

plot_tuning_signatures(tun_vectors[1] * 2, fig_size=(2, 2), dpi=200)

#%%

def print_morph_space(data, title=None):
    morph_space_data = np.reshape(data, [25, 150, -1])
    print("shape morph_space_data", np.shape(morph_space_data))

    # fig, axs = plt.subplots(len(morph_space_data))
    # for i in range(len(morph_space_data)):
    #     axs[i].plot(morph_space_data[i])

    # get max values for each video and category
    amax_ms = np.amax(morph_space_data, axis=1)
    print("shape amax_ms", np.shape(amax_ms))
    print(amax_ms)

    # make into grid
    amax_ms_grid = np.reshape(amax_ms, [5, 5, -1])
    print("shape amax_ms_grid", np.shape(amax_ms_grid))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(amax_ms_grid[..., 1], cmap='hot', interpolation='nearest')
    axs[0, 1].imshow(amax_ms_grid[..., 2], cmap='hot', interpolation='nearest')
    axs[1, 0].imshow(amax_ms_grid[..., 3], cmap='hot', interpolation='nearest')
    axs[1, 1].imshow(amax_ms_grid[..., 4], cmap='hot', interpolation='nearest')
    plt.show()


print_morph_space(NRE_proj[:3750], title="Human")

