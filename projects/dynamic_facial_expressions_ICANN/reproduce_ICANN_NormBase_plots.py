"""
2020/12/11
This script plots_utils the results from reproduce_ICANN_NormBase.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.load_config import load_config

# t0001: 2-norm     t0002: 1-norm   t0003: simplified   t0004: direction-only   t0005: expressitivity-direction
# t0006: 2-norm-monkey_morph

config = load_config("norm_base_reproduce_ICANN_t0006.json")
save_name = config["sub_folder"]
save_folder = os.path.join("../../models/saved", config['save_name'], save_name)
accuracy = np.load(os.path.join(save_folder, "accuracy.npy"))
it_resp = np.load(os.path.join(save_folder, "it_resp.npy"))
labels = np.load(os.path.join(save_folder, "labels.npy"))

print("accuracy", accuracy)
print("it_resp.shape", it_resp.shape)
print("labels.shape", labels.shape)

# colors 0=Neutral=Black, 1=Threat=Yellow, 2=Fear=Blue, 3=LipSmacking=Red
colors = config['colors']
titles = config['condition']
seq_length = config['seq_length']
n_cat = config['n_category']
labels = config['labels']

# data loaded is the concatenation of train_data and test_data
# loop over all condition to sort it_resp per condition (minus neutral)
concat_seq_start = np.array(config['concat_seq_start'])
shape_n_seq = np.shape(concat_seq_start)
n_sequence = shape_n_seq[1]
n_condition = shape_n_seq[0]
sorted_it_resp = np.zeros((n_condition, n_sequence, seq_length, n_cat))
for i in range(n_condition):
    for j in range(n_sequence):
        sorted_it_resp[i, j] = it_resp[concat_seq_start[i, j]:concat_seq_start[i, j]+seq_length]
print("shape sorted_it_resp", np.shape(sorted_it_resp))

# plot all it responses for one stimulus
fig = plt.figure(figsize=(15,10))
plt.subplots(n_condition,1)
plt.suptitle("Face Neuron Responses")
for i in range(n_condition):  # n_condition (i.e. anger fear lip_smack)
    plt.subplot(n_condition,1,i+1)
    for j in range(n_sequence):  # n_sequences  (i.e. 25-50-75-100)
        plt.plot(range(seq_length), sorted_it_resp[i, j, :, i+1], color=colors[i+1])
    plt.ylabel(titles[i+1])
plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig3A.png"))

# plot it_resp over stimulus intensity
fig = plt.figure(figsize=(15,10))
plt.subplots(n_condition,1)
plt.suptitle("Face Neuron Responses")
for i in range(n_condition):  # n_condition (anger fear lip_smack)
    plt.subplot(n_condition,1,i+1)
    plt.ylabel(titles[i+1])
    for j in range(n_sequence):  # n_sequences  (i.e. 25-50-75-100)
        plt.plot(range(seq_length),sorted_it_resp[i, j, :, i+1],
                 label=labels[j],
                 color=colors[i+1],
                 linewidth=n_sequence/(n_sequence-j))
    plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig4A.png"))

# plot activity in function of expressivity level
fig = plt.figure(figsize=(15,10))
plt.subplots(n_condition,1)
plt.suptitle("Expression Neuron Responses")
for i in range(n_condition):  # n_condition (anger fear lip_smack)
    plt.subplot(n_condition,1,i+1)
    plt.ylabel(titles[i + 1])
    maximums = np.amax(sorted_it_resp[i, :, :, i+1], axis=1)
    max_normed = maximums / np.amax(maximums)
    plt.plot(range(n_sequence), max_normed, color=colors[i+1], label=titles[i+1])
    plt.xticks(config['xticks'])
    plt.xlabel(labels)
    plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig4B.png"))