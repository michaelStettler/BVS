"""
2020/12/11
This script plots the results from reproduce_ICANN.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.load_config import load_config

# t0001: 2-norm     t0002: 1-norm   t0003: simplified   t0004: direction-only   t0005: expressitivity-direction

config = load_config("norm_base_reproduce_ICANN_t0005.json")
save_name = config["sub_folder"]
save_folder = os.path.join("models/saved", config['save_name'], save_name)
accuracy = np.load(os.path.join(save_folder, "accuracy.npy"))
it_resp = np.load(os.path.join(save_folder, "it_resp.npy"))
labels = np.load(os.path.join(save_folder, "labels.npy"))

print("accuracy", accuracy)
print("it_resp.shape", it_resp.shape)
print("labels.shape", labels.shape)

# data loaded is the concatenation of train_data and test_data
# --> process data
# Fear 1.0 in 0:120
# LipSmacking 1.0 in 120:240
# Threat 1.0 in 240:360
# Fear 0.25 in 360:480
# Fear 0.5 in 480:600
# Fear 0.75 in 600:720
# LipSmacking 0.25 in 720:840
# LipSmacking 0.5 in 840:960
# LipSmacking 0.75 in 960:1080
# Threat 0.25 in 1080:1200
# Threat 0.5 in 1200:1320
# Threat 0.75 in 1320:1440
# it_resp 0=Neutral, 1=Threat, 2=Fear, 3=LipSmacking
it_resp_threat = [it_resp[240:360,:], it_resp[1320:1440,:],it_resp[1200:1320,:], it_resp[1080:1200,:]]
it_resp_fear = [it_resp[0:120,:], it_resp[600:720,:],it_resp[480:600,:], it_resp[360:480,:]]
it_resp_lipSmacking = [it_resp[120:240,:], it_resp[960:1080,:],it_resp[840:960,:], it_resp[720:840,:]]
# colors 0=Neutral=Black, 1=Threat=Yellow, 2=Fear=Blue, 3=LipSmacking=Red
colors = ['k', 'y', 'b', 'r']
titles = ['Neutral','Threat', 'Fear', 'LipSmacking']

# plot it_resp over stimulus intensity
fig = plt.figure(figsize=(15,10))
plt.subplots(3,1)
plt.suptitle("Face Neuron Response")
for i, data in enumerate([it_resp_threat, it_resp_fear, it_resp_lipSmacking]):
    plt.subplot(3,1,i+1)
    plt.ylabel(titles[i+1])
    for j, percent in enumerate(['100%', '75%', '50%', '25%']):
        plt.plot(range(120),data[j][:,i+1], label=percent, color=colors[i+1], linewidth=4/(j+1))
    plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig4.png"))

# plot all it responses for one stimulus
fig = plt.figure(figsize=(15,10))
plt.subplots(3,1)
plt.suptitle("Face Neuron Response")
for i, data in enumerate([it_resp_threat, it_resp_fear, it_resp_lipSmacking]):
    plt.subplot(3,1,i+1)
    for j in range(4):
        plt.plot(range(120), data[0][:,j], color=colors[j], label=titles[j])
    plt.ylabel(titles[i+1])
plt.legend()
plt.savefig(os.path.join(save_folder, "plot_ICANN_Fig3.png"))