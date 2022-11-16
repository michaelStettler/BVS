import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.01_morph_space_with_NRE
"""

#%% declare script variables
show_plot = False
load_RBF_pattern = False
train_RBF_pattern = True
save_RBF_pattern = True
load_FR_pathway = True
save_FR_pos = False
load_FER_pos = False
save_FER_pos = True
save_FER_with_lmk_name = True

#%% declare hyper parameters
n_iter = 2
max_sigma = None
max_sigma = 1000
train_idx = None
# train_idx = [50]

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
config["FER_lmk_name"] = ["left_eyelid"]
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
print("len FR_patterns_list[1]", len(FR_patterns_list[1]))
print("len FER_patterns_list", len(FER_patterns_list))
print("len FER_sigma_list", len(FER_sigma_list))
print("shape FER_patterns_list", np.shape(FER_patterns_list))
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
    print("load FER pos")
    FER_pos = np.load(os.path.join(config["directory"], config["LMK_data_directory"], "FER_LMK_pos.npy"))
else:
    print("create FER pos")
    FER_pos = create_lmk_dataset(train_data[0], v4_model, "FER", config, FER_patterns_list, FER_sigma_list)

    if save_FER_pos:
        if save_FER_with_lmk_name:
            np.save(os.path.join(config["directory"], config["LMK_data_directory"],
                                 "FER_LMK_pos" + "_" + config["FER_lmk_name"][0]), FER_pos)

            FER_pos = merge_LMK_pos(config)
        else:
            np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FER_LMK_pos"), FER_pos)
print("shape FER_pos", np.shape(FER_pos))
print()

#%%
# plot_sequence(train_data[0], lmks=FER_pos*4, pre_processing='VGG19')


#%% learn reference vector
ref_idx = [0, 3750]
ref_idx = [0]
avatar_labels = np.array([0, 1]).astype(int)
avatar_labels = np.array([0]).astype(int)
# ref_vectors = learn_ref_vector(FER_pos[ref_idx], train_data[1][ref_idx], avatar_labels=avatar_labels, n_avatar=2)
ref_vectors = learn_ref_vector(FER_pos[ref_idx], train_data[1][ref_idx], avatar_labels=avatar_labels, n_avatar=1)
print("shape ref_vectors", np.shape(ref_vectors))

#%%
plot_sequence(train_data[0], lmks=FER_pos*4, ref_lmks=ref_vectors*4, pre_processing='VGG19', lmk_size=3)
print("sequence created")

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
tun_idx = [0] + get_morph_extremes_idx()[:4]
tun_vectors = learn_tun_vectors(FER_pos[tun_idx], train_data[1][tun_idx], ref_vectors, face_ids[tun_idx], n_cat=5)
tun_vectors[4, 9, 1] = 0
print("shape tun_vectors", np.shape(tun_vectors))
print(tun_vectors)

#%% Compute projections
# todo remove face positions from the FER_pos
# NRE_proj = compute_projections(FER_pos, face_ids, ref_vectors, tun_vectors, return_proj_length=True)
(NRE_proj, NRE_proj_lmk) = compute_projections(FER_pos[:3750], face_ids[:3750], ref_vectors, tun_vectors,
                                               neutral_threshold=0,
                                               return_proj_length=True,
                                               return_proj_lmks=True)
print("shape NRE_proj", np.shape(NRE_proj))

# #%%
# for i in range(5, 10):
#     print("shape NRE_proj[:150]", np.shape(NRE_proj[:150]))
#     print("max NRE_proj[:150]", np.amax(NRE_proj[150*i:150*(i+1)], axis=0))
#     # plot test human fear
#     plt.figure()
#     plt.plot(NRE_proj[150*i:150*(i+1)])
#     plt.legend(["N", "HA", "HF", "MA", "MF"])
#     plt.title("seq_{}".format(0))
#     plt.show()

#%%
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tqdm
from plots_utils.plot_ft_map_pos import _set_fig_name
from plots_utils.plot_ft_map_pos import _set_save_folder


def get_tuning_plot(ref_vectors, tun_vectors, lmks, colors, proj=None, lmk_proj=None, fig_title=None, fig_size=(2, 2), dpi=200):
    # print("shape ref_vectors", np.shape(ref_vectors))
    # print("shape projections", np.shape(projections))
    # print("shape tun_vectors", np.shape(tun_vectors))
    # create figure
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    # center ref_vectors
    centers = np.mean(ref_vectors, axis=0)
    ref_vect = ref_vectors - centers
    lmks_vect = lmks - centers

    for i, ref_v, tun_v, lmk, color in zip(np.arange(len(ref_vectors)), ref_vect, tun_vectors, lmks_vect, colors):
        plt.scatter(ref_v[0], -ref_v[1], color=color, facecolors='none')
        plt.arrow(ref_v[0], -ref_v[1], tun_v[0], -tun_v[1], color=color, linewidth=1, linestyle=':')  # reference

        plt.scatter(lmk[0], -lmk[1], color=color, marker='x')

        if lmk_proj is not None:
            plt.text(lmk[0] + .5, -lmk[1] + .5, "{:.2f}".format(lmk_proj[i]), fontsize="xx-small")

    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.axis('off')

    if proj is not None:
        plt.text(-5, 18, "{:.2f}".format(proj))  # text in data coordinates

    if fig_title is not None:
        plt.title(fig_title)

    # transform figure to numpy array
    fig.canvas.draw()
    fig.tight_layout(pad=0)

    figure = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # clear figure
    plt.cla()
    plt.clf()
    plt.close()

    return figure


def get_image_sequence(image, lmks, ref_lmks, colors, lmk_size=3):
    lmks = lmks.astype(int)
    ref_lmks = ref_lmks.astype(int)

    # compute padding
    lmk_pad = int(lmk_size / 2)

    # construct image of the sequence
    # add lmk on image
    for l, lmk, r_lmk, color in zip(range(len(lmks)), lmks, ref_lmks, colors):

        # add ref lmk on image
        image[r_lmk[1] - lmk_pad:r_lmk[1] + lmk_pad, r_lmk[0] - lmk_pad:r_lmk[0] + lmk_pad] = [255, 0, 0]

        # add arrows
        color = np.array(color)[:3] * 255
        image = cv2.arrowedLine(image, (r_lmk[0], r_lmk[1]), (lmk[0], lmk[1]), color, 1)

        # add lmks
        image[lmk[1] - lmk_pad:lmk[1] + lmk_pad, lmk[0] - lmk_pad:lmk[0] + lmk_pad] = [0, 255, 0]

    return image


def get_graph(data, frame, fig_size=(4, 1), dpi=200):
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    plt.plot([frame, frame], [0, 10], 'r-')
    plt.plot(data)

    fig.canvas.draw()
    # fig.tight_layout(pad=0.5)

    figure = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # clear figure
    plt.cla()
    plt.clf()
    plt.close()

    return figure


def plot_signature_proj_sequence(data, lmks, ref_vectors, tun_vectors, proj, lmk_proj=None, pre_processing=None, video_name=None, path=None, lmk_size=3):
    print("shape data", np.shape(data))
    print("shape tun_vectors", np.shape(tun_vectors))
    print("shape lmks", np.shape(lmks))
    print("shape proj", np.shape(proj))

    img_seq_ratio = 4

    if pre_processing == 'VGG19':
        data_copy = []
        for image in data:
            # (un-)process image from VGG19 pre-processing
            img = np.array(image + 128).astype(np.uint8)
            # img = img[..., ::-1]  # rgb
            img[img > 255] = 255
            img[img < 0] = 0
            data_copy.append(img)
        data = np.array(data_copy)

    # retrieve parameters
    img_seq_height = 1000
    img_seq_width = 1600
    img_seq = np.zeros((img_seq_height, img_seq_width, 3)).astype(np.uint8)
    print("shape img_seq", np.shape(img_seq))

    # set name
    if video_name is not None:
        video_name = video_name
    else:
        video_name = 'video.mp4'

    # set path
    if path is not None:
        path = path
    else:
        path = ''

    # create colors
    colors = cm.rainbow(np.linspace(0, 1, np.shape(lmks)[1]))

    # set video recorder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(path, video_name), fourcc, 30, (img_seq_width, img_seq_height))

    # set conditions
    conditions = ["Hum. Anger", "Hum. Fear", "Monk. Anger", "Monk. Fear"]
    x_pos = [0, 0, 1, 1]
    y_pos = [0, 1, 0, 1]

    # construct movie
    for i, image in tqdm.tqdm(enumerate(data)):
        img = get_image_sequence(image, lmks[i]*img_seq_ratio, ref_vectors[0]*img_seq_ratio, colors,
                                 lmk_size=lmk_size)
        img = cv2.resize(img, (800, 800))
        img_text = "Frame {}, Seq. {}".format(i, int(i/150))
        img = cv2.putText(img, img_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        img_seq[100:900, :800] = img

        # construct signatures
        for c, cond in enumerate(conditions):
            lmk_p_ = None
            if lmk_proj is not None:
                lmk_p_ = lmk_proj[i, c + 1]

            fig = get_tuning_plot(ref_vectors[0], tun_vectors[c + 1] * 4, lmks[i], colors,
                                  proj=proj[i, c + 1],
                                  lmk_proj=lmk_p_,
                                  fig_title=cond,
                                  fig_size=(2, 2),
                                  dpi=200)  # construct fig of 400x400
            img_seq[(x_pos[c]*400):(x_pos[c]*400+400), (y_pos[c]*400+800):(y_pos[c]*400+1200)] = fig

        # add graphics
        graph = get_graph(proj, i)
        img_seq[800:, 800:] = graph

        # write image
        video.write(img_seq)

    cv2.destroyAllWindows()
    video.release()


indexes = [np.arange(150), np.arange(750, 900), np.arange(3600, 3750)]
video_names = ["HA_Angry_1.0_Human_1.0.mp4", "HA_Angry_1.0_Human_0.75.mp4", "HA_Angry_0.0_Human_0.0.mp4"]

for index, video_name in zip(indexes, video_names):
    plot_signature_proj_sequence(np.array(train_data[0][index]), FER_pos[index], ref_vectors, tun_vectors, NRE_proj[index],
                                 video_name=video_name,
                                 lmk_proj=NRE_proj_lmk[index],
                                 pre_processing='VGG19',
                                 lmk_size=3)
print("finish creating sequence")

matplotlib.use('macosx')
# plt.figure()
# plt.imshow(fig)

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

    cat_grid = np.zeros((5, 5, 5))
    prob_grid = np.zeros((5, 5, 5))
    for i in range(np.shape(amax_ms_grid)[0]):
        for j in range(np.shape(amax_ms_grid)[0]):
            x = amax_ms_grid[i, j]
            cat_grid[i, j, np.argmax(x)] = 1
            prob_grid[i, j] = np.exp(x)/sum(np.exp(x))

    # print category grid
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(cat_grid[..., 1], cmap='hot', interpolation='nearest')
    axs[0, 1].imshow(cat_grid[..., 2], cmap='hot', interpolation='nearest')
    axs[1, 0].imshow(cat_grid[..., 3], cmap='hot', interpolation='nearest')
    axs[1, 1].imshow(cat_grid[..., 4], cmap='hot', interpolation='nearest')
    plt.show()

    # print probability grid
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(prob_grid[..., 1], cmap='hot', interpolation='nearest')
    axs[0, 1].imshow(prob_grid[..., 2], cmap='hot', interpolation='nearest')
    axs[1, 0].imshow(prob_grid[..., 3], cmap='hot', interpolation='nearest')
    axs[1, 1].imshow(prob_grid[..., 4], cmap='hot', interpolation='nearest')
    plt.show()


print_morph_space(NRE_proj[:3750], title="Human")

