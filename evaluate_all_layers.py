import json
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.load_data import load_data
from utils.data_generator import DataGen
from utils.load_model import load_model

from models.NormBase import NormBase

config_path = 'configs/norm_base_config'
# config_name = 'norm_base_monkey_test.json'
config_name = 'norm_base_affectNet_sub8_4000_t0005.json'
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

print("[LOOP] calculate for {}".format(v4_layers))

accuracies = np.zeros(len(v4_layers))
for i_layer, layer in enumerate(v4_layers):
    config['v4_layer'] = layer
    print('[LOOP] start with v4_layer: {}'.format(config['v4_layer']))

    print("[LOOP] start training")

    # create model
    norm_base = NormBase(config, input_shape=(224, 224, 3))

    if not os.path.join("models/saved", config['save_name']):
        os.mkdir(os.path.join("models/saved", config['save_name']))
    # folder for save and load
    save_folder = os.path.join("models/saved", config['save_name'], config['v4_layer'])
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    try:
        # load vectors if available
        ref_vector = np.load(os.path.join(save_folder, "ref_vector.npy"))
        tun_vector = np.load(os.path.join(save_folder, "tuning_vector.npy"))
        print("[MODEL] ref_vector and tun_vector are available and loaded from {}"
              .format(save_folder))

        norm_base.set_ref_vector(ref_vector)
        norm_base.set_tuning_vector(tun_vector)
        print("[MODEL] Set ref vector", np.shape(ref_vector))
        print("[MODEL] Set tuning vector", np.shape(tun_vector))
    except IOError:
        # calculate vectors if not available
        # load train data
        data_train = load_data(config)
        print("[Data] -- Data loaded --")

        # train model
        ref_vector, tun_vector = norm_base.fit(data_train, batch_size=config['batch_size'])

        # save model
        np.save(os.path.join(save_folder, "ref_vector"), ref_vector)
        np.save(os.path.join(save_folder, "tuning_vector"), tun_vector)

    print("[LOOP] start prediction")

    # load test data
    data_test = load_data(config, train=False, sort_by=['image'])
    print("[Data] -- Data loaded --")

    try:
        # load results if available
        accuracy = np.load(os.path.join(save_folder, "accuracy.npy"))
        it_resp = np.load(os.path.join(save_folder, "it_resp.npy"))
        labels = np.load(os.path.join(save_folder, "labels.npy"))
        print("[MODEL] it_resp is available and is loaded from {}".format(save_folder))
    except IOError:
        #calculate results if not available
        #evaluate
        accuracy, it_resp, labels = norm_base.evaluate_accuracy(data_test)
        np.save(os.path.join(save_folder, "accuracy"), accuracy)
        np.save(os.path.join(save_folder, "it_resp"), it_resp)
        np.save(os.path.join(save_folder, "labels"), labels)

    print("accuracy", accuracy)
    print("shape it_resp", np.shape(it_resp))
    print("shape labels", np.shape(labels))

    accuracies[i_layer] = accuracy

    print('[LOOP] finished with v4_layer: {}'.format(config['v4_layer']))

#plt.plot(np.arange(len(accuracies)), accuracies)
print(accuracies)
print(np.argmax(accuracies))
plt.plot(v4_layers, accuracies)
plt.xticks(rotation=90)
plt.savefig(os.path.join("models/saved", config['save_name'], "plot_accuracy_pool.png"))
