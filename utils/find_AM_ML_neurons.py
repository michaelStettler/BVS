import json
import numpy as np
import tensorflow as tf

from utils.load_data import load_data
from utils.load_model import load_model


def find_SA_tuning(model, img, features):
    """
    Implementation of the method "Shape-appearance tuning experiment" (SA)from the paper :
    'Convolutional neural networks explain tuning properties of anterior, but not middle, face-processing areas in
    macaque inferotemporal cortex'
    That computes the Shape preference index (SPI) AS: SPI = (S - A) / (S + A) with A the 25 shape dimensions vector and
    S the 25 appearance dimensions vector.

    :param model:
    :param data:
    :return:
    """
    # todo set dictionary of face selective units

    print("shape img", np.shape(img))
    print("shape features", np.shape(features))

    SPI = []
    for layer in model.layers[:2]:
        if "conv" in layer.name:
            print("layer:", layer.name)

            # cut model
            m = tf.keras.Model(inputs=model.input, outputs=layer.output)

            # predict face and non_face outputs
            preds = m.predict(img)

            #  compute the average response for each unit
            a = np.reshape(preds, (np.shape(preds)[0], -1))

            # compute STA following Doris Paper: 'The code for facial identity in the primate brain' formula 2
            # A = sum(s(n)*f(n)) / sum(f(n))
            # f(n):= the spiking response -> activations (a) units
            # s(n):= feature vector
            STA = features.T @ a
            num = STA[:25, :] - STA[25:, :]
            denom = STA[:25, :] + STA[25:, :]

            # compute SPI by removing the ones that are equal to 0
            SPIs = num
            SPIs[num != 0] = num[num != 0] / denom[denom != 0]
            layer_SPI = np.mean(SPIs, axis=0)

            # save SPI
            SPI.append(layer_SPI)

    return SPI


if __name__ == "__main__":
    import os
    config_file_path = 'configs/face_units/find_SA_units_mac.json'
    save = True

    np.set_printoptions(precision=3, suppress=True, linewidth=150)

    # load find_face config
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    # load model
    model = load_model(config)
    # print(model.summary())

    # load data
    data = load_data(config)
    img = data[0]
    features = data[1]

    # compute face units
    SA_units = find_SA_tuning(model, img, features)
    print("Shape face_units", np.shape(SA_units))