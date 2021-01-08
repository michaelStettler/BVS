import os
import numpy as np
import matplotlib.pyplot as plt
from utils.load_config import load_config
from models.NormBase import NormBase
from utils.load_data import load_data

config = load_config("norm_base_plotDirections_t0009.json")
save_name = config["sub_folder"]
retrain = False

# model
try:
    if retrain:
        raise IOError("retrain = True")
    norm_base = NormBase(config, input_shape=(224, 224, 3), save_name=save_name)
except IOError:
    norm_base = NormBase(config, input_shape=(224,224,3))

    norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][0]), fit_dim_red=True, fit_ref=False,
                  fit_tun=False)
    norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][1]), fit_dim_red=False, fit_ref=True,
                  fit_tun=False)
    norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][2]), fit_dim_red=False, fit_ref=False,
                  fit_tun=True)
    norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][3]), fit_dim_red=False, fit_ref=True,
                  fit_tun=False)
    norm_base.save_model(config, save_name)

# test
data_test = load_data(config, train=config["data_test"])

projection, labels = norm_base.projection_tuning(data_test)

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
    x_max = max(max(np.abs(ax.get_xlim())), 1)
    ax.set_xlim(xmin=-x_max, xmax=x_max)

    #plot lines of constant activation
    ax.plot(x_lines, lines, color="k", linewidth=0.5)
plt.savefig(os.path.join("models/saved", config['save_name'], save_name, "scatter.png"))