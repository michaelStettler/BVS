import json
import numpy as np
import tensorflow as tf

from utils.load_data import load_data
from utils.load_model import load_model


def find_face_units(model, data, verbose=False):
    """
    Implementation of the method "Face-selective population estimation" from the paper :
    ""Convolutional neural networks explain tuning properties of anterior, but not middle, face-processing areas in
    macaque inferotemporal cortex

    :param model:
    :param config:
    :return:
    """

    print("shape data", np.shape(data))
    x_face = data[:50]
    x_object = data[50:]
    FSI_list = []
    for layer in model.layers:
        if "conv" in layer.name:
            if verbose:
                print("layer:", layer.name)

            # cut model
            m = tf.keras.Model(inputs=model.input, outputs=layer.output)

            # predict face and non_face outputs
            preds_face = m.predict(x_face)
            preds_object = m.predict(x_object)

            # flatten array
            preds_face = np.reshape(preds_face, (np.shape(preds_face)[0], -1))
            preds_object = np.reshape(preds_object, (np.shape(preds_object)[0], -1))
            n_features = np.shape(preds_face)[1]

            # compute average response R_face and R_object
            r_face = np.mean(preds_face, axis=0)
            r_object = np.mean(preds_object, axis=0)

            # compute FSI
            nume = r_face - r_object
            denom = r_face + r_object
            FSI = nume
            # remove case where denom could be equal to zero
            FSI[denom != 0] = nume[denom != 0] / denom[denom != 0]

            # set FSI to 1 or -1 in function of R_face and R_object
            for i in range(len(FSI)):
                if r_face[i] > 0 > r_object[i]:
                    FSI[i] = 1
                elif r_face[i] < 0 < r_object[i]:
                    FSI[i] = -1

            # save index and values of face units
            idx = np.arange(len(FSI))
            FSI_idx = idx[FSI > 1/3]
            FSI_val = FSI[FSI_idx]
            FSI_list.append([FSI_idx, FSI_val])

            if verbose:
                print("found:", len(FSI_idx), "face units ({:.2f}%)".format(len(FSI_idx)/n_features * 100))

    return np.array(FSI_list)


if __name__ == "__main__":
    import os
    config_file_path = 'configs/face_units/find_face_units_test_mac.json'
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
    x = data[0]
    y = data[1]

    # compute face units
    face_units = find_face_units(model, x, verbose=True)
    print("Shape face_units", np.shape(face_units))

    # save face units
    if save:
        np.save(os.path.join(config['save_path'], config['model']), face_units)

    # todo make a fice-selective feature map ?

