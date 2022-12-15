import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from datasets_utils.morphing_space import get_morph_extremes_idx
from datasets_utils.morphing_space import get_NRE_from_morph_space
from datasets_utils.merge_LMK_pos import merge_LMK_pos

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
from plots_utils.plot_sequence import plot_sequence
from plots_utils.plot_signature_analysis import plot_signature_proj_analysis

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
save_FER_with_lmk_name = True
load_ref = True
save_ref = False
load_tun = True
save_tun = False
# norm_type = 'individual'
norm_type = 'categorical'
# occluded and orignial are the same for this pipeline as we do not have any landmark on the ears
condition = ["human_orig", "monkey_orig", "human_equi", "monkey_equi"]
train_csv = ["/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/morphing_space_human_orig_train.csv",
             "/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/morphing_space_monkey_orig_train.csv",
             "/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/morphing_space_human_equi_train.csv",
             "/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/morphing_space_monkey_equi_train.csv"]
cond = 2

#%% declare hyper parameters
n_iter = 2
max_sigma = None
max_sigma = 3000
train_idx = None
# train_idx = [50]

#%% import config
config_path = 'BH_01_morph_space_with_NRE_m0001.json'
# load config
config = load_config(config_path, path='configs/behavourial')

# edit dictionary for single condition type
if cond is not None:
    config["train_csv"] = train_csv[cond]
    config["condition"] = condition[cond]
    if "human" in condition[cond]:
        config["avatar_types"] = ["human"]
    else:
        config["avatar_types"] = ["monkey"]

# create directory
save_path = os.path.join(config["directory"], config["LMK_data_directory"], config["condition"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

print("-- Config loaded --")
print()

config["FR_lmk_name"] = ["left_eye", "right_eye", "nose"]
config["FR_lmk_name"] = []

config["FER_lmk_name"] = ["left_eyebrow_ext", "left_eyebrow_int", "right_eyebrow_int", "right_eyebrow_ext",
                 "left_mouth", "top_mouth", "right_mouth", "down_mouth",
                 "left_eyelid", "right_eyelid"]
config["FER_lmk_name"] = ["right_eyelid"]
# config["FER_lmk_name"] = []


#%% import data
train_data = load_data(config)
print("-- Data loaded --")
print("len train_data[0]", len(train_data[0]))
print()

#%% split training for LMK and norm base
NRE_train = get_NRE_from_morph_space(train_data, condition=condition[cond])
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
    print("len FR_patterns_list[0]", len(FR_patterns_list[0]))
    print("len FR_patterns_list[1]", len(FR_patterns_list[1]))
    print("len FER_patterns_list", len(FER_patterns_list))
    print("len FER_sigma_list", len(FER_sigma_list))
    print("shape FER_patterns_list", np.shape(FER_patterns_list))
    print()

if train_RBF_pattern:
    print("create patterns")
    print("shape FER_patterns_list", np.shape(FER_patterns_list))
    FR_patterns_list, FR_sigma_list, FER_patterns_list, FER_sigma_list = \
        create_RBF_LMK(config, LMK_train, v4_model,
                       max_sigma=max_sigma,
                       n_iter=n_iter,
                       FR_patterns=FR_patterns_list,
                       FR_sigma=FR_sigma_list,
                       FER_patterns=FER_patterns_list,
                       FER_sigma=FER_sigma_list,
                       save=save_RBF_pattern,
                       train_idx=train_idx)

    print("len FR_patterns_list", len(FR_patterns_list))
    print("len FR_patterns_list[0]", len(FR_patterns_list[0]))
    print("len FER_patterns_list", len(FER_patterns_list))
    print("len FER_sigma_list", len(FER_sigma_list))
    print("shape FER_patterns_list", np.shape(FER_patterns_list))
    print()

#%% get identity and positions from the FR Pathway
extremes_idx = get_morph_extremes_idx()
if load_FR_pathway:
    FR_pos = np.load(os.path.join(save_path, "FR_LMK_pos.npy"))
    face_ids = np.load(os.path.join(save_path, "face_identities.npy"))
    face_positions = np.load(os.path.join(save_path, "face_positions.npy"))
else:
    FR_pos, face_ids, face_positions = get_identity_and_pos(train_data[0], v4_model, config, FR_patterns_list, FR_sigma_list)

    if save_FR_pos:
        np.save(os.path.join(save_path, "FR_LMK_pos"), FR_pos)
        np.save(os.path.join(save_path, "face_positions"), face_positions)
        np.save(os.path.join(save_path, "face_identities"), face_ids)
print("shape FR_pos", np.shape(FR_pos))
print("shape face_ids", np.shape(face_ids))
print("face_ids[0]", face_ids[0])
print("shape face_positions", np.shape(face_positions))
print()

#%% transform face_idx to a array of zeros since we have only one condition at time
face_ids = np.zeros(np.shape(face_ids)).astype(int)
print("shape face_ids", np.shape(face_ids))
print("face_ids[0]", face_ids[0])


#%% predict LMK pos
if load_FER_pos:
    print("load FER pos")
    if os.path.exists(os.path.join(save_path, "FER_LMK_pos.npy")):
        FER_pos = np.load(os.path.join(save_path, "FER_LMK_pos.npy"))
    else:
        FER_pos = merge_LMK_pos(config)
else:
    print("create FER pos")
    FER_pos = create_lmk_dataset(train_data[0], v4_model, "FER", config, FER_patterns_list, FER_sigma_list)

    if save_FER_pos:
        if save_FER_with_lmk_name:
            np.save(os.path.join(save_path, "FER_LMK_pos" + "_" + config["FER_lmk_name"][0]), FER_pos)

            FER_pos = merge_LMK_pos(config)
        else:
            np.save(os.path.join(save_path, "FER_LMK_pos"), FER_pos)
print("shape FER_pos", np.shape(FER_pos))
print()

#%%
# plot_sequence(train_data[0], lmks=FER_pos*4, pre_processing='VGG19')


#%% learn reference vector
ref_idx = [0]
avatar_labels = np.array([0]).astype(int)

if load_ref:
    ref_vectors = np.load(os.path.join(save_path, "ref_vectors.npy"))
else:
    # ref_vectors = learn_ref_vector(FER_pos[ref_idx], train_data[1][ref_idx], avatar_labels=avatar_labels, n_avatar=2)
    ref_vectors = learn_ref_vector(FER_pos[ref_idx], train_data[1][ref_idx], avatar_labels=avatar_labels, n_avatar=1)

    if save_ref:
        np.save(os.path.join(save_path, "ref_vectors"), ref_vectors)

print("shape ref_vectors", np.shape(ref_vectors))

#%%
# plot_sequence(train_data[0], lmks=FER_pos*4, ref_lmks=ref_vectors*4, pre_processing='VGG19', lmk_size=3)
# print("sequence created")

#%% plot landmarks on NRE_train
if show_plot:
    extremes_idx = get_morph_extremes_idx()
    # extremes_idx = [0] + extremes_idx[:4] + [3750] + extremes_idx[4:]
    extremes_idx = [0] + extremes_idx[:4]
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
tun_idx = [0] + get_morph_extremes_idx(config["condition"])[:4]
if load_tun:
    tun_vectors = np.load(os.path.join(save_path, "tun_vectors.npy"))
else:
    tun_vectors = learn_tun_vectors(FER_pos[tun_idx], train_data[1][tun_idx], ref_vectors, face_ids[tun_idx], n_cat=5)

    if save_tun:
        np.save(os.path.join(save_path, "tun_vectors"), tun_vectors)

print("shape tun_vectors", np.shape(tun_vectors))
print(tun_vectors)
print()

#%% Compute projections
print("compute projections")
# todo remove face positions from the FER_pos
(NRE_proj, NRE_proj_lmk) = compute_projections(FER_pos, face_ids, ref_vectors, tun_vectors,
                                               norm_type=norm_type,
                                               neutral_threshold=0,
                                               return_proj_length=True,
                                               return_proj_lmks=True)
print("shape NRE_proj", np.shape(NRE_proj))
print()

# #%% plot categorization over sequence
# for i in range(5, 10):
#     print("shape NRE_proj[:150]", np.shape(NRE_proj[:150]))
#     print("max NRE_proj[:150]", np.amax(NRE_proj[150*i:150*(i+1)], axis=0))
#     # plot test human fear
#     plt.figure()
#     plt.plot(NRE_proj[150*i:150*(i+1)])
#     plt.legend(["N", "HA", "HF", "MA", "MF"])
#     plt.title("seq_{}".format(0))
#     plt.show()

#%% plot sequence analysis
indexes = [np.arange(150), np.arange(750, 900), np.arange(3600, 3750)]
video_names = ["HA_Angry_1.0_Human_1.0.mp4", "HA_Angry_1.0_Human_0.75.mp4", "HA_Angry_0.0_Human_0.0.mp4"]

for index, video_name in zip(indexes, video_names):
    plot_signature_proj_analysis(np.array(train_data[0][index]), FER_pos[index], ref_vectors, tun_vectors,
                                 NRE_proj[index], config,
                                 video_name=video_name,
                                 lmk_proj=NRE_proj_lmk[index],
                                 pre_processing='VGG19',
                                 lmk_size=3)
print("finish creating sequence analysis")

matplotlib.use('macosx')


#%%
def print_morph_space(amax_ms_grid=None, cat_grid=None, prob_grid=None,
                      title=None, show_plot=True, save=True, save_path=None):
    if amax_ms_grid is not None:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(amax_ms_grid[..., 1], cmap='hot', interpolation='nearest')
        axs[0, 1].imshow(amax_ms_grid[..., 2], cmap='hot', interpolation='nearest')
        axs[1, 0].imshow(amax_ms_grid[..., 3], cmap='hot', interpolation='nearest')
        axs[1, 1].imshow(amax_ms_grid[..., 4], cmap='hot', interpolation='nearest')

        if show_plot:
            plt.show()
        if save:
            if save_path is None:
                plt.savefig("morph_space_read_out_values_{}.jpeg".format(norm_type))
            else:
                plt.savefig(os.path.join(save_path, "morph_space_read_out_values_{}.jpeg".format(norm_type)))

    if cat_grid is not None:
        # print category grid
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(cat_grid[..., 1], cmap='hot', interpolation='nearest')
        axs[0, 1].imshow(cat_grid[..., 2], cmap='hot', interpolation='nearest')
        axs[1, 0].imshow(cat_grid[..., 3], cmap='hot', interpolation='nearest')
        axs[1, 1].imshow(cat_grid[..., 4], cmap='hot', interpolation='nearest')

        if show_plot:
            plt.show()
        if save:
            if save_path is None:
                plt.savefig("morph_space_categories_values_{}.jpeg".format(norm_type))
            else:
                plt.savefig(os.path.join(save_path, "morph_space_categories_values_{}.jpeg".format(norm_type)))

    # print probability grid
    if cat_grid is not None:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(prob_grid[..., 1], cmap='hot', interpolation='nearest')
        axs[0, 1].imshow(prob_grid[..., 2], cmap='hot', interpolation='nearest')
        axs[1, 0].imshow(prob_grid[..., 3], cmap='hot', interpolation='nearest')
        axs[1, 1].imshow(prob_grid[..., 4], cmap='hot', interpolation='nearest')

        if show_plot:
            plt.show()
        if save:
            if save_path is None:
                plt.savefig("morph_space_probabilities_values_{}.jpeg".format(norm_type))
            else:
                plt.savefig(os.path.join(save_path, "morph_space_probabilities_values_{}.jpeg".format(norm_type)))

morph_space_data = np.reshape(NRE_proj[:3750], [25, 150, -1])
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

cat_grid = np.zeros((5, 5, 5))
prob_grid = np.zeros((5, 5, 5))
for i in range(np.shape(amax_ms_grid)[0]):
    for j in range(np.shape(amax_ms_grid)[0]):
        x = amax_ms_grid[i, j]
        cat_grid[i, j, np.argmax(x)] = 1
        prob_grid[i, j] = np.exp(x) / sum(np.exp(x))

np.save(os.path.join(save_path, "amax_ms_grid_{}".format(norm_type)), amax_ms_grid)
np.save(os.path.join(save_path, "cat_grid_{}".format(norm_type)), cat_grid)
np.save(os.path.join(save_path, "prob_grid_{}".format(norm_type)), prob_grid)


print_morph_space(amax_ms_grid, cat_grid, prob_grid, show_plot=True, title="Human")

