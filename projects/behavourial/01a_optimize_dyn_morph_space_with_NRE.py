import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from datasets_utils.morphing_space import get_morph_extremes_idx
from datasets_utils.morphing_space import get_extrm_frame_from_morph_space
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
from projects.behavourial.project_utils import *


from plots_utils.plot_BVS import display_images
from plots_utils.plot_sequence import plot_sequence
from plots_utils.plot_signature_analysis import plot_signature_proj_analysis

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.01a_optimize_dyn_morph_space_with_NRE
"""

#%% declare script variables
computer = 'mac'
computer_path, computer_letter = get_computer_path(computer)

show_plot = False
model_name = 'NRE'
# norm_type = 'individual'
norm_type = 'frobenius'
# occluded and orignial are the same for this pipeline as we do not have any landmark on the ears
conditions = ["human_orig", "monkey_orig"]
cond = 0
condition = conditions[cond]
train_csv = [os.path.join(computer_path, "morphing_space_human_orig.csv"),
             os.path.join(computer_path, "morphing_space_monkey_orig.csv")]
modality = 'dynamic'
prot_indexes = [np.arange(150), np.arange(600, 750), np.arange(3000, 3150), np.arange(3600, 3750)]

#%% declare hyper parameters
taus_u = [1, 2, 3, 4, 5, 6]
taus_y = [1, 1.5, 2, 2.5, 3, 3.5]
taus_d = [1, 2, 3, 4, 5, 6]

#%% import config
config_path = 'BH_01_morph_space_with_NRE_{}0001.json'.format(computer_letter)
# load config
config = load_config(config_path, path='configs/behavourial')

# edit dictionary for single condition type
if cond is not None:
    config["train_csv"] = train_csv[cond]
    config["condition"] = condition
    if "human" in condition:
        config["avatar_types"] = ["human"]
    else:
        config["avatar_types"] = ["monkey"]

# create directory
load_path = os.path.join(config["directory"], config["LMK_data_directory"], condition)
save_lmk_path = load_path
save_path = os.path.join(config["directory"], config["save_path"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

print("-- Config loaded --")
print()


#%% import data
train_data = load_data(config)
morph_space = "/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace"
behav_path = "morphing_psychophysics_result"
if condition == "human_orig":
    behav_data = np.load(os.path.join(morph_space, behav_path, "human_avatar_orig.npy"))
elif condition == "monkey_orig":
    behav_data = np.load(os.path.join(morph_space, behav_path, "monkey_avatar_orig.npy"))

behav_data = np.moveaxis(behav_data, 0, -1)
print("-- Data loaded --")
print("len train_data[0]", len(train_data[0]))
print("shape behav_data", np.shape(behav_data))
print()

#%% split training for LMK and norm base
NRE_train = get_extrm_frame_from_morph_space(train_data, condition=condition)
LMK_train = train_data  # take all

print("-- Data Split --")
print("len NRE_train[0]", len(NRE_train[0]))
print("NRE_train[1]")
print(NRE_train[1])
print()

#%% get identity and positions from the FR Pathway
FR_pos = np.load(os.path.join(load_path, "FR_LMK_pos.npy"))
face_ids = np.load(os.path.join(load_path, "face_identities.npy"))
face_positions = np.load(os.path.join(load_path, "face_positions.npy"))

if 'human' in condition:
    FR_pos = FR_pos[:3750]
    face_ids = face_ids[:3750]
    face_positions = face_positions[:3750]
else:
    FR_pos = FR_pos[3750:]
    face_ids = face_ids[3750:]
    face_positions = face_positions[3750:]

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
print("load FER pos")
if os.path.exists(os.path.join(load_path, "FER_LMK_pos.npy")):
    FER_pos = np.load(os.path.join(load_path, "FER_LMK_pos.npy"))
else:
    FER_pos = merge_LMK_pos(config)

if 'human' in condition:
    FER_pos = FER_pos[:3750]
else:
    FER_pos = FER_pos[3750:]

#%% get reference vector
ref_vectors = np.load(os.path.join(load_path, "ref_vectors.npy"))
print("shape ref_vectors", np.shape(ref_vectors))

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
tun_vectors = np.load(os.path.join(load_path, "tun_vectors.npy"))

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


#%% Compute differentiators
def compute_dynamic_responses(seq_resp, n_cat=5, tau_u=3, tau_v=3, tau_y=15, tau_d=5, w_inhib=0.8, get_differentiator=False):
    """
    Compute the dynamic responses of recognition neurons
    Compute first a differentiation circuit followed by a competitive network
    :param seq_resp:
    :return:
    """
    seq_length = np.shape(seq_resp)[0]

    # --------------------------------------------------------------------------------------------------------------
    # compute differentitator

    # declare differentiator
    v_df = np.zeros((seq_length, n_cat))  # raw difference
    pos_df = np.zeros((seq_length, n_cat))  # positive flanks
    neg_df = np.zeros((seq_length, n_cat))  # negative flanks
    y_df = np.zeros((seq_length, n_cat))  # integrator

    for f in range(1, seq_length):
        # compute differences
        pos_dif = seq_resp[f - 1] - v_df[f - 1]
        pos_dif[pos_dif < 0] = 0
        neg_dif = v_df[f - 1] - seq_resp[f - 1]
        neg_dif[neg_dif < 0] = 0

        # update differentiator states
        v_df[f] = ((tau_v - 1) * v_df[f - 1] + seq_resp[f - 1]) / tau_v
        pos_df[f] = ((tau_u - 1) * pos_df[f - 1] + pos_dif) / tau_u
        neg_df[f] = ((tau_u - 1) * neg_df[f - 1] + neg_dif) / tau_u
        y_df[f] = ((tau_y - 1) * y_df[f - 1] + pos_df[f - 1] + neg_df[f - 1]) / tau_y

    # --------------------------------------------------------------------------------------------------------------
    # compute decision network

    # declare inhibition kernel
    inhib_k = (1 - np.eye(n_cat) * 0.8) * w_inhib
    # declare decision neurons
    ds_neuron = np.zeros((seq_length, n_cat))

    for f in range(1, seq_length):
        # update decision neurons
        ds_neur = ((tau_d - 1) * ds_neuron[f - 1] + y_df[f - 1] - inhib_k @ ds_neuron[f - 1]) / tau_d

        # apply activation to decision neuron
        ds_neur[ds_neur < 0] = 0
        ds_neuron[f] = ds_neur

    if get_differentiator:
        return ds_neuron, np.array([pos_df, neg_df])
    else:
        return ds_neuron


#%% get decision neurons
def get_morph_space_dyn(NRE_proj, tau_u, tau_v, tau_d, tau_y):
    ds_neurons = []
    for i in range(25):
        ds_neuron = compute_dynamic_responses(NRE_proj[i*150:i*150+150],
                                                       tau_u=tau_u,
                                                       tau_v=tau_v,
                                                       tau_d=tau_d,
                                                       tau_y=tau_y)
        ds_neurons.append(ds_neuron)
    ds_neurons = np.reshape(ds_neurons, (-1, np.shape(ds_neurons)[-1]))

    morph_space_data = np.reshape(ds_neurons[:3750], [25, 150, -1])

    return morph_space_data

#%%
def KL_divergence(p, q):
    return np.sum(p * np.log(p / q))


def compute_morph_space_KL_div(p, q):
    dim_x = np.shape(p)[0]
    dim_y = np.shape(p)[1]

    divergences = np.zeros((dim_x, dim_y))
    for x in range(dim_x):
        for y in range(dim_y):
            div = KL_divergence(p[x, y], q[x, y])
            divergences[x, y] = div

    return divergences

#%% compute all kl_div (grid search)
kl_divergences = np.zeros((len(taus_u), len(taus_y), len(taus_d)))
for u, tau_u in enumerate(taus_u):
    for y, tau_y in enumerate(taus_y):
        for d, tau_d in enumerate(taus_d):
            # get morph space
            ms_dyn = get_morph_space_dyn(NRE_proj, tau_u, tau_u, tau_d, tau_y)

            # get max over sequence
            ms_dyn = np.amax(ms_dyn, axis=1)
            ms_dyn = np.reshape(ms_dyn, [5, 5, -1])

            # transform to morph space prob
            ms_prob_grid = np.zeros((5, 5, 4))
            for i in range(5):
                for j in range(5):
                    x = ms_dyn[i, j, 1:]  # discard neutral
                    ms_prob_grid[i, j] = np.exp(x) / sum(np.exp(x))

            # get KL div
            kl_divs = compute_morph_space_KL_div(behav_data, ms_prob_grid)

            # save divergence
            sum_KL_div = np.sum(kl_divs)
            kl_divergences[u, y, d] = sum_KL_div
            print(f"tau_u/v: {tau_u}, tau_y: {tau_y}, tau_d: {tau_d}, KL: {sum_KL_div}")

kl_divergences = np.array(kl_divergences)
print("amin kl_divergences", np.amin(kl_divergences))
print("amax kl_divergences", np.amax(kl_divergences))
arg_min_idx = np.unravel_index(np.argmin(kl_divergences, axis=None), kl_divergences.shape)
print(f"arg_min_idx: {arg_min_idx}")
print(f"tau_u: {taus_u[arg_min_idx[0]]} tau_y: {taus_y[arg_min_idx[1]]} tau_d: {taus_d[arg_min_idx[2]]}")
