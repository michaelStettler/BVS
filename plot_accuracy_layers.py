import json
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.load_model import load_model

from utils.load_model import load_model

config_path = 'configs/norm_base_config'
config_name = 'norm_base_affectNet_sub8_4000_t0004.json'
config_file_path = os.path.join(config_path, config_name)
print("config_file_path", config_file_path)
# load norm_base_config file
with open(config_file_path) as json_file:
    config = json.load(json_file)

if config['v4_layer'] == "all":
    v4_layers = []
    model = load_model(config, input_shape=(224, 224, 3))
    for layer in model.layers[1:]:
        v4_layers.append(layer.name)
elif isinstance(config['v4_layer'],list):
    v4_layers = config['v4_layer']
else:
    raise ValueError("v4_layer: {} is chosen, but should be a list! Please choose [\"layer1\", \"layer2\"] instead!"
                     .format(config['v4_layer']))

accuracies = np.zeros(len(v4_layers))
for i_layer, layer in enumerate(v4_layers):
    config['v4_layer'] = layer

    # folder for load
    load_folder = os.path.join("models/saved", config['save_name'], config['v4_layer'])
    accuracy = np.load(os.path.join(load_folder, "accuracy.npy"))
    accuracies[i_layer] = accuracy

# print maximum
print("max accuracy", np.max(accuracies))
print("max layer no", np.argmax(accuracies))
print("max layer", v4_layers[np.argmax(accuracies)])

#create plot
fig = plt.figure(figsize=(30,15))
ax = plt.axes()
plt.title("Accuracy over Layers")
plt.plot(v4_layers, accuracies)
plt.xticks(rotation=90)
ax.xaxis.grid()

#save plot
plt.savefig(os.path.join("models/saved", config['save_name'], "plot_accuracy_pool.png"))
