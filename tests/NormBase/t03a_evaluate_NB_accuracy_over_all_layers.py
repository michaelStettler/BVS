import tensorflow as tf
import os
import numpy as np

from utils.load_model import load_model
from utils.load_config import load_config
from utils.evaluate_model import evaluate_model


def evaluate_all_layers(config):
    if config['v4_layer'] == "all":
        v4_layers = []
        model = load_model(config, input_shape=(224, 224, 3))
        for layer in model.layers[1:]:
            v4_layers.append(layer.name)
        tf.keras.backend.clear_session()
    elif isinstance(config['v4_layer'],list):
        v4_layers = config['v4_layer']
    else:
        raise ValueError("v4_layer: {} is chosen, but should be a list! Please choose [\"layer1\", \"layer2\"] instead!"
                         .format(config['v4_layer']))

    print("[LOOP] calculate for {}".format(v4_layers))

    for i_layer, layer in enumerate(v4_layers):
        config['v4_layer'] = layer
        print('[LOOP] start with v4_layer: {}'.format(config['v4_layer']))
        print('[LOOP] layer %i of %i' % (i_layer+1, len(v4_layers)))

        accuracy, it_resp, labels, ref_vector, tun_vector = evaluate_model(config, config['v4_layer'], legacy=True)
        tf.keras.backend.clear_session()

        print("accuracy", accuracy)
        print("shape it_resp", np.shape(it_resp))
        print("shape labels", np.shape(labels))

        print('[LOOP] finished with v4_layer: {}'.format(config['v4_layer']))


if __name__ == "__main__":

    """
    2020/12/03
    This script evaluates the accuracy over all layers. Results are plotted with t03b_plot_accuracy_layers.py
    2021/01/18
    Note: norm_base_affectNet_sub8_4000_t0005.json does not use dimensionality reduction. 
    Therefore, the accuracy is very bad.
    """

    config_names = ['norm_base_affectNet_sub8_4000_t0005.json']
    for config_name in config_names:
        config = load_config(config_name)
        evaluate_all_layers(config)