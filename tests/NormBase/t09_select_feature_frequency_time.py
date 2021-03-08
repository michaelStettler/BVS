"""
2021/3/5
This script demonstrates how to select feature (in this case positions) based on their frequency in time.
2 plots are created:
- power spectrum
- positions video with feature maps highlighted where x and y are selected

Future use:
- use selection_array to reduce the array of positions (recommended by Tim)
- use highlight_intersect to reduce the feature maps
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase
from utils.plot_cnn_output import plot_cnn_output
from utils.calculate_position import calculate_position

# load config
config = load_config("norm_base_config_plot_frequency_time_t0001.json")

# load data --> 150 x 224 x 224 x 3
images,_ = load_data(config, train=1)

# load NormBase
norm_base = NormBase(config, (224,224,3))

# (train NormBase)--> maybe necessary, but not atm

# get_preds --> 150 x 512, x and y concatenated
positions = norm_base.get_preds(images)

# calculate fft over time and leave out constant term
rfft = np.fft.rfft(positions, axis=0)
rfft_abs = np.abs(rfft)
power_spectrum = np.square(rfft_abs)[1:]
#calculate frequencies and leave out constant term
frequencies = np.fft.rfftfreq(positions.shape[0], d=1/30)[1:]
print("frequencies", frequencies)

# plot fft discarding constant offset (at zero frequency)
fig = plt.figure()
plt.semilogy(frequencies, power_spectrum)
plt.ylim(1e-6, np.max(power_spectrum))
fig.savefig(os.path.join("models/saved", config["save_name"], config["plot_name"]))

# calculate mean freq
#mean_freq = np.sum(energy * freq) / np.sum(energy)
frequencies_mean = np.einsum("ij,i->j", power_spectrum, frequencies) / np.sum(power_spectrum, axis=0)
#--> produces some nan values but these are discarded by sorting later
print("len(frequencies_mean)", len(frequencies_mean))

# select lowest frequencies
n_filter_time = 256
features_sorted_freq = np.argsort(frequencies_mean)
selection_array = np.zeros(len(frequencies_mean), dtype=bool)
selection_array[features_sorted_freq[:n_filter_time]] = True
print("selection_array", selection_array)

# calculate selected positions
selection_indices = np.arange(len(selection_array))[selection_array]
print("selected positions", selection_indices)

# calculate selected feature map depending on whether x and y component from this feature map is selected
highlight_x = selection_indices[selection_indices<256]
highlight_y = selection_indices[selection_indices>256] % 256
highlight_union = list(set(highlight_x) | set(highlight_y))
highlight_intersect = list(set(highlight_x) & set(highlight_y))
print("len(highlight_union)", len(highlight_union))
print("len(highlight_intersect)", len(highlight_intersect))

# animate positions and highlight selected ones
positions_array = calculate_position( norm_base.evaluate_v4(images, flatten=False),
    mode=config["position_method"], return_mode="array")
plot_cnn_output(positions_array, os.path.join("models/saved/", config["save_name"]), config["plot_name"]+"_video.mp4",
                title=None, image=images, video=True, highlight=highlight_intersect)