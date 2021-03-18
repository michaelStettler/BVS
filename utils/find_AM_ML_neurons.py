import json
import numpy as np
import tensorflow as tf

from utils.load_data import load_data
from utils.load_extraction_model import load_model


def find_SA_tuning(model, img, features, face_units=None, verbose=False):
    """
    Implementation of the method "Shape-appearance tuning experiment" (SA)from the paper :
    'Convolutional neural networks explain tuning properties of anterior, but not middle, face-processing areas in
    macaque inferotemporal cortex'
    That computes the Shape preference index (SPI) AS: SPI = (S - A) / (S + A) with A the 25 shape dimensions vector and
    S the 25 appearance dimensions vector.

    :param model:
    :param img:
    :param features: (50,) dimensional vector that generated the image
    :param face_idx:
    :return:
    """

    print("shape img", np.shape(img))
    print("shape features", np.shape(features))

    SPIs = []
    l = 0  # cannot use the enumerate fct since we are passing some layers as we keep only the "conv"
    for layer in model.layers:
        if "conv" in layer.name:
            if verbose:
                print("layer:", layer.name)

            # cut model
            m = tf.keras.Model(inputs=model.input, outputs=layer.output)

            # predict face and non_face outputs
            preds = m.predict(img)

            #  flatten answers
            a = np.reshape(preds, (np.shape(preds)[0], -1))

            # if a face units dict is present, keep only these units
            if face_units is not None:
                a = a[:, face_units[l]]

            # compute STA following Doris Paper: 'The code for facial identity in the primate brain' formula 2
            # A = sum(s(n)*f(n)) / sum(f(n))
            # f(n):= the spiking response -> activations (a) units
            # s(n):= feature vector

            # note: I remove all activations that are equal to 0 all the time
            norm = np.sum(a, axis=0)
            print("shape a", np.shape(a))
            a = a[:, norm != 0]
            print("shape a", np.shape(a))
            norm = norm[norm != 0]

            #  compute STA
            STA = (features.T @ a) / norm

            # compute SPI by removing the ones that are equal to 0
            S = np.linalg.norm(STA[:25], axis=0)
            A = np.linalg.norm(STA[25:], axis=0)
            num = S - A
            denom = S + A
            SPI = num / denom

            SPIs.append(SPI)

            if verbose:
                print("shape SPI", np.shape(SPI))
                print("min max SPI", np.amin(SPI), np.amax(SPI))
                print()

            # increase index
            l += 1

    return SPIs


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

    # load face_units
    face_units = np.load(os.path.join(config['face_units_path'], config['model']+'.npy'), allow_pickle=True)
    face_units = face_units[:, 0]  # keep only the idx part

    # compute face units
    SA_units = find_SA_tuning(model, img, features, face_units=face_units, verbose=True)
    print("Shape face_units", np.shape(SA_units))

    # save AM/ML units
    if save:
        np.save(os.path.join(config['save_path'], config['model']), SA_units)