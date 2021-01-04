import os
import numpy as np
import matplotlib.pyplot as plt
from utils.load_config import load_config
from models.NormBase import NormBase
from utils.load_data import load_data

config = load_config("norm_base_plotDirections_t0001.json")
save_name = "humanAvatar"
retrain = False

# model
try:
    if retrain:
        raise IOError("retrain = True")
    norm_base = NormBase(config, input_shape=(224, 224, 3), save_name=save_name)
except IOError:
    norm_base = NormBase(config, input_shape=(224,224,3))
    data_train = load_data(config)
    norm_base.fit(data_train)
    norm_base.save_model(config, save_name)

# test
data = load_data(config, train=False)
projection, labels = norm_base.projection_tuning(data)

# calculate constant activation lines
x_lines, lines = norm_base.line_constant_activation()

# plot
fig, axs = plt.subplots(1, projection.shape[0], figsize=(5*projection.shape[0], 5))
fig.suptitle("Projection of difference vector keeping 2-norm and scalar product")
for category, ax in enumerate(axs):
    ax.set_title('category {}'.format(category))
    ax.scatter(projection[category,:,0], projection[category,:,1], s = 1, c=labels)

    # set axis limits
    ax.set_ylim(ymin=0, ymax=ax.get_ylim()[1]*1.2)
    x_max = max(np.abs(ax.get_xlim()))
    ax.set_xlim(xmin=-x_max, xmax=x_max)

    #plot lines of constant activation
    ax.plot(x_lines, lines, color="k", linewidth=0.5)
plt.savefig(os.path.join("models/saved", config['save_name'], save_name, "scatter.png"))