import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_SA_units(SA_units, layer_names, max_columns=4):
    # get number of panels (= num of layers)
    num_panels = len(SA_units)
    num_rows = int(np.ceil(num_panels / max_columns))

    # set figure
    fig, axs = plt.subplots(num_rows, max_columns)

    # for each layer
    for l, SA_unit in enumerate(SA_units):
        row = int(l / max_columns)
        col = l % max_columns

        axs[row, col].hist(SA_unit, bins=np.arange(-1, 1, .05))
        axs[row, col].set_title(layer_names[l])


if __name__ == "__main__":
    from utils.load_model import load_model

    config_file_path = 'configs/face_units/find_SA_units_mac.json'

    # load find_face config
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    model = load_model(config)
    # print(model.summary())

    # get model layer name
    layer_names = []
    for layer in model.layers:
        if "conv" in layer.name:
            layer_names.append(layer.name)

    # get SA_units
    SA_units = np.load(os.path.join(config['save_path'], config['model']+'.npy'), allow_pickle=True)
    print("shape face units", np.shape(SA_units))

    plot_SA_units(SA_units, layer_names)

    plt.show()