import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load_model import load_model

config_file_path = 'configs/face_units/find_semantic_units_test_mac.json'

# load find_face config
with open(config_file_path) as json_file:
    config = json.load(json_file)

model = load_model(config)
print(model.summary())

# get model layer name
layer_names = []
for layer in model.layers:
    if "conv" in layer.name:
        layer_names.append(layer.name)

# get face_units
face_units = np.load(os.path.join(config['save_path'], config['model']+'.npy'), allow_pickle=True)
print("shape face units", np.shape(face_units))
print("shape face units[:, 0]", np.shape(face_units[:, 0]))

# count number of face units per layer
count_units = []
for f, face_unit in enumerate(face_units[:, 0]):
    count_units.append(len(face_unit))
x = np.arange(len(count_units))
print(count_units)

# create histogram
g = sns.barplot(x=x, y=count_units)
g.set_title(config['model'])
g.set_xticks(x)
# g.set_xticklabels(layer_names, rotation=45, horizontalalignment='right')
g.set_xticklabels(layer_names, rotation=90)
g.set_xlabel("Layer name")
g.set_ylabel("# face units")

# display counting values
for i in x:
    if count_units[i] > 0:
        g.text(i, count_units[i] + 0.2, str(count_units[i]), color='black', ha="center")

plt.show()
