import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from datasets_utils.expressivity_level import get_expr_extreme_idx
from datasets_utils.expressivity_level import get_extreme_frame_from_expr_strength
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
run: python -m projects.behavourial.04_expression_strength
"""
#%% declare script variables
show_plot = True
load_RBF_pattern = True
train_RBF_pattern = False
save_RBF_pattern = False
load_FR_pathway = True
save_FR_pos = False
load_FER_pos = True
save_FER_pos = False
save_FER_with_lmk_name = False
load_ref = True
save_ref = False
load_tun = True
save_tun = True
load_test_lmk_pos = True
save_test_lmk_pos = False
# norm_type = 'individual'
# norm_type = 'categorical'  # deprecated
norm_type = 'frobenius'
# occluded and orignial are the same for this pipeline as we do not have any landmark on the ears
train_csv = ["/Users/michaelstettler/PycharmProjects/BVS/data/ExpressivityLevels/train.csv"]
avatar_name = ["monkey"]

#%% declare hyper parameters
n_iter = 2
max_sigma = None
max_sigma = 3000
train_idx = None

#%% import config
config_path = 'BH_04_expr_strength_m0001.json'
# load config
config = load_config(config_path, path='configs/behavourial')

# create directory
save_path = os.path.join(config["directory"], config["LMK_data_directory"])
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_path = os.path.join(save_path, config["condition"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

print("-- Config loaded --")
print()

config["FR_lmk_name"] = ["left_eye", "right_eye", "nose"]
# config["FR_lmk_name"] = []

config["FER_lmk_name"] = ["left_eyebrow_ext", "left_eyebrow_int", "right_eyebrow_int", "right_eyebrow_ext",
                 "left_mouth", "top_mouth", "right_mouth", "down_mouth",
                 "left_eyelid", "right_eyelid"]
# config["FER_lmk_name"] = ["right_eyelid"]
# config["FER_lmk_name"] = []

#%% import data
train_data = load_data(config)
print("-- Data loaded --")
print("len train_data[0]", len(train_data[0]))
print("size train_data[0]", np.shape(train_data[0]))
print()

# crop images
# train_data[0] = train_data[0][]

#%% split training for LMK and norm base
NRE_train = get_extreme_frame_from_expr_strength(train_data)
LMK_train = train_data  # take all

print("-- Data Split --")
print("len NRE_train[0]", len(NRE_train[0]))
print("NRE_train[1]")
print(NRE_train[1])
print()

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
        load_RBF_patterns_and_sigma(config, avatar_name=avatar_name)
    print("len FR_patterns_list[0]", len(FR_patterns_list[0]))
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
extremes_idx = get_expr_extreme_idx()
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

#%% plot landmarks on NRE_train
if show_plot:
    extremes_idx = get_expr_extreme_idx()
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
tun_idx = [0] + get_expr_extreme_idx()
if load_tun:
    tun_vectors = np.load(os.path.join(save_path, "tun_vectors.npy"))
else:
    tun_vectors = learn_tun_vectors(FER_pos[tun_idx], train_data[1][tun_idx], ref_vectors, face_ids[tun_idx],
                                    n_cat=config["n_category"])

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

#%% plot projections for each sequence
if show_plot:
    for i in range(config["n_category"] - 1):  # don't plot neutral
        seq_idx = i * 120
        print("shape NRE_proj[:120]", np.shape(NRE_proj[seq_idx:seq_idx+120]))
        print("max NRE_proj[:120]", np.amax(NRE_proj[seq_idx:seq_idx+120], axis=0))
        # plot test human fear
        plt.figure()
        plt.plot(NRE_proj[seq_idx:seq_idx+120])
        plt.legend(["N", "Fear", "Lipsmack", "Threat"])
        plt.title("seq_{}".format(0))
        plt.show()

#%% plot sequence analysis
if show_plot:
    indexes = [np.arange(120), np.arange(120, 240), np.arange(240, 360)]
    video_names = ["Fear.mp4", "Lipsmack.mp4", "Threat.mp4"]

    for index, video_name in zip(indexes, video_names):
        print("shape train_data[0][index]", np.shape(train_data[0][index]))
        plot_signature_proj_analysis(np.array(train_data[0][index]), FER_pos[index], ref_vectors, tun_vectors,
                                     NRE_proj[index], config,
                                     video_name=video_name,
                                     lmk_proj=NRE_proj_lmk[index],
                                     pre_processing='VGG19',
                                     lmk_size=3)
    print("finish creating sequence analysis")

    matplotlib.use('macosx')

#%% Load all dataset for predictions
print("----------------------------------")
print("-----------  TEST  ---------------")
test_data = load_data(config, train=False)
print("len test_data[0]", len(test_data[0]))

if load_test_lmk_pos:
    FER_test_pos = np.load(os.path.join(save_path, "FER_LMK_test_pos.npy"))
else:
    print("create FER pos")
    FER_test_pos = create_lmk_dataset(test_data[0], v4_model, "FER", config, FER_patterns_list, FER_sigma_list)

    if save_test_lmk_pos:
        np.save(os.path.join(save_path, "FER_LMK_test_pos"), FER_test_pos)
print("shape FER_test_pos", np.shape(FER_test_pos))

# %% Compute projections
print("compute projections")
test_face_ids = np.zeros(len(test_data[0])).astype(int)  # just do this because there's only 1 monkey avatar...
(NRE_test_proj, NRE_proj_test_lmk) = compute_projections(FER_test_pos, test_face_ids, ref_vectors, tun_vectors,
                                                         norm_type=norm_type,
                                                         neutral_threshold=0,
                                                         return_proj_length=True,
                                                         return_proj_lmks=True)
print("shape NRE_test_proj", np.shape(NRE_test_proj))
print()

#%% plot sequences plot
if show_plot:
    fig, axs = plt.subplots(3)
    colors = np.array([(0, 0, 255), (0, 191, 0), (237, 0, 0)]) / 255
    linewidths = [0.5, 1.0, 1.5, 2.0]
    fig.suptitle('NRE Predictions for expression strength level')
    for i in range(3):
        seq_idx = (i * 4 + 3) * 120  # +00% is at pos 4
        max_data = np.amax(NRE_test_proj[seq_idx:seq_idx + 120, i+1])

        for j in range(4):  # don't plot neutral
            seq_idx = (i * 4 + j) * 120

            seq_data = NRE_test_proj[seq_idx:seq_idx + 120, i+1] / max_data
            axs[i].plot(seq_data, color=colors[i], linewidth=linewidths[j])
            if i == 0:
                axs[i].legend(["25", "50", "75", "100"])
    plt.savefig(f"NRE_{norm_type}_projections.svg", format='svg')
    plt.show()

#%% bar plot
if show_plot:
    fig, axs = plt.subplots(1, 3)
    colors = np.array([(0, 0, 255), (0, 191, 0), (237, 0, 0)]) / 255
    titles = ["Fear", "Lipsmack", "threat"]
    # fig.suptitle('NRE Predictions for expression strength level')
    for i in range(3):
        heights = []
        for j in range(4):  # don't plot neutral
            seq_idx = (i * 4 + j) * 120
            seq = [NRE_test_proj[seq_idx:seq_idx + 120, i+1]]
            heights.append(np.amax(seq))

        x = [0, 1, 2, 3]
        y = heights/np.amax(heights)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        axs[i].bar(x, y, color=colors[i])
        axs[i].set_xticks(np.arange(4), ['25', '50', '75', '100'])
        axs[i].set_title(titles[i], color=colors[i])
        axs[i].plot(x, p(x), color='black', linewidth=2)

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        if i > 0:
            axs[i].spines['left'].set_visible(False)
            axs[i].get_yaxis().set_ticks([])
        # axs[i].spines['bottom'].set_visible(False)
    plt.savefig(f"NRE_{norm_type}_bar_expr_level.svg", format='svg')
    plt.show()



