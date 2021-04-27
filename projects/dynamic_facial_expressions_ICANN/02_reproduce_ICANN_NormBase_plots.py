"""
2020/12/11
This script plots_utils the results from 01_reproduce_ICANN_NormBase.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.load_config import load_config
from models.NormBase import NormBase

# t0001: 2-norm     t0002: 1-norm   t0003: simplified   t0004: direction-only   t0005: expressitivity-direction
# t0006: 2-norm-monkey_morph

do_reverse = False
do_normalize = False

config = load_config("norm_base_reproduce_ICANN_t0015.json", path="configs/norm_base_config")
save_name = config["sub_folder"]
save_folder = os.path.join("models/saved", config['save_name'], save_name)
accuracy = np.load(os.path.join(save_folder, "accuracy.npy"))
it_resp = np.load(os.path.join(save_folder, "it_resp.npy"))
labels = np.load(os.path.join(save_folder, "labels.npy"))
norm_base = NormBase(config, input_shape=(224,224,3))


print("accuracy", accuracy)
print("it_resp.shape", it_resp.shape)
print("labels.shape", labels.shape)

# colors 0=Neutral=Black, 1=Threat=Yellow, 2=Fear=Blue, 3=LipSmacking=Red
colors = config['colors']
titles = config['condition']
seq_length = config['seq_length']
n_cat = config['n_category']
labels = config['labels']

if config['use_ggplot'] is not None:
    use_ggplot = config['use_ggplot']
    if use_ggplot:
        import seaborn as sns
        plt.style.use('seaborn-paper')

if config['show_legends'] is not None:
    show_legends = config['show_legends']

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------  sort data for plotting -----------------------------------------------------------------
# data loaded is the concatenation of train_data and test_data
# loop over all condition to sort it_resp per condition (minus neutral)
concat_seq_start = np.array(config['concat_seq_start'])
shape_n_seq = np.shape(concat_seq_start)
n_sequence = shape_n_seq[1]
n_condition = shape_n_seq[0]
sorted_it_resp = np.zeros((n_condition, n_sequence, seq_length, n_cat))
sorted_expression_resp = np.zeros((n_condition, n_sequence, seq_length, n_cat))
for i in range(n_condition):
    for j in range(n_sequence):
        seq = it_resp[concat_seq_start[i, j]:concat_seq_start[i, j]+seq_length]

        # recreate the flipped results, since norm base is static, we can simply flip the IT response
        if do_reverse:
            seq = np.flip(seq, axis=0)

        sorted_it_resp[i, j] = seq
        expression_resp = norm_base.compute_dynamic_responses(seq)
        sorted_expression_resp[i, j] = expression_resp
print("shape sorted_it_resp (n_condition, n_sequence, length_seq, n_category)", np.shape(sorted_it_resp))

if do_normalize:
    # normalize data
    # max it
    max_it_resp = np.amax(sorted_it_resp, axis=(1, 2, 3))
    max_it_resp = np.expand_dims(max_it_resp, axis=1)
    max_it_resp = np.repeat(max_it_resp, np.shape(sorted_it_resp)[1], axis=1)
    max_it_resp = np.expand_dims(max_it_resp, axis=2)
    max_it_resp = np.repeat(max_it_resp, np.shape(sorted_it_resp)[2], axis=2)
    max_it_resp = np.expand_dims(max_it_resp, axis=3)
    max_it_resp = np.repeat(max_it_resp, np.shape(sorted_it_resp)[3], axis=3)
    sorted_it_resp /= max_it_resp
    # max expression
    max_expr_resp = np.amax(sorted_expression_resp, axis=(1, 2, 3))
    max_expr_resp = np.expand_dims(max_expr_resp, axis=1)
    max_expr_resp = np.repeat(max_expr_resp, np.shape(sorted_expression_resp)[1], axis=1)
    max_expr_resp = np.expand_dims(max_expr_resp, axis=2)
    max_expr_resp = np.repeat(max_expr_resp, np.shape(sorted_expression_resp)[2], axis=2)
    max_expr_resp = np.expand_dims(max_expr_resp, axis=3)
    max_expr_resp = np.repeat(max_expr_resp, np.shape(sorted_expression_resp)[3], axis=3)
    sorted_expression_resp /= max_expr_resp
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------  plotting  ------------------------------------------------------------------------------
# plot all it responses for one stimulus
fig = plt.figure(figsize=(15, 10))
plt.subplots(n_condition, 1)
plt.suptitle("Face Neuron Responses")
for i in range(n_condition):  # n_condition (i.e. anger fear lip_smack)
    plt.subplot(n_condition, 1, i+1)
    for j in range(n_cat):
        plt.plot(range(seq_length), sorted_it_resp[i, n_sequence - 1, :, j],
                 color=colors[j],
                 label=titles[j],
                 linewidth=2)

    plt.ylabel(titles[i+1])
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    if do_normalize:
        plt.ylim([-0.05, 1.1])
plt.xlabel("frame")
if show_legends:
    plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig3A.png"))

# plot all expression neurons responses for one stimulus
fig = plt.figure(figsize=(15,10))
plt.subplots(n_condition,1)
plt.suptitle("Expression Neuron Responses")
for i in range(n_condition):  # n_condition (i.e. anger fear lip_smack)
    plt.subplot(n_condition, 1, i+1)
    for j in range(n_cat):
        plt.plot(range(seq_length), sorted_expression_resp[i, n_sequence - 1, :, j],
                 color=colors[j],
                 label=titles[j],
                 linewidth=2)

    plt.ylabel(titles[i+1])
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    if do_normalize:
        plt.ylim([-0.05, 1.1])
plt.xlabel("frame")
if show_legends:
    plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig3B.png"))

# plot it_resp over stimulus intensity
fig = plt.figure(figsize=(15,10))
plt.subplots(n_condition,1)
plt.suptitle("Face Neuron Responses")
for i in range(n_condition):  # n_condition (anger fear lip_smack)
    plt.subplot(n_condition,1,i+1)
    for j in range(n_sequence):  # n_sequences  (i.e. 25-50-75-100)
        plt.plot(range(seq_length),sorted_it_resp[i, j, :, i+1],
                 label=labels[j],
                 color=colors[i+1],
                 linewidth=n_sequence/(n_sequence-j))

    plt.ylabel(titles[i+1])
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    if do_normalize:
        plt.ylim([-0.05, 1.1])
plt.xlabel("frame")
if show_legends:
        plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig4A.png"))

# plot activity in function of expressivity level
fig = plt.figure(figsize=(15,10))
plt.subplots(n_condition, 1)
plt.suptitle("Expression Neuron Responses")
for i in range(n_condition):  # n_condition (anger fear lip_smack)
    plt.subplot(n_condition, 1, i+1)
    maximums = np.amax(sorted_it_resp[i, :, :, i+1], axis=1)
    maximums = np.sum(sorted_it_resp[i, :, :, i+1], axis=1)
    max_normed = maximums / np.amax(maximums)
    plt.plot(config['xticks'], max_normed,
             color=colors[i+1],
             label=titles[i+1],
             linewidth=2)

    plt.ylabel(titles[i + 1])
    # plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    if do_normalize:
        plt.ylim([-0.05, 1.1])
    plt.xticks(config['xticks'], labels)
plt.xlabel("frame")
if show_legends:
    plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig4B.png"))