import json
import numpy as np
import tensorflow as tf

from utils.load_data import load_data
from utils.load_model import load_model


def found_face_units(model, config):
    """
    Implement the method "Face-selective population estimation" from the paper :
    ""Convolutional neural networks explain tuning properties of anterior, but not middle, face-processing areas in macaque inferotemporal cortex

    :param model:
    :param config:
    :return:
    """
    data = load_data(config)
    x = data[0]
    y = data[1]

    x_face = x[:50]
    x_object = x[50:]
    print("shape x", np.shape(x))
    FSI_list = []
    for layer in model.layers:
        if "conv" in layer.name:
            print("layer:", layer.name)

            # cut model
            m = tf.keras.Model(inputs=model.input, outputs=layer.output)

            # predict face and non_face outputs
            preds_face = m.predict(x_face)
            preds_object = m.predict(x_object)

            # compute average response R_face and R_object
            r_face = np.mean(preds_face, axis=(0, 1, 2))
            r_object = np.mean(preds_object, axis=(0, 1, 2))

            # compute FSI
            FSI = (r_face - r_object) / (r_face + r_object)

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

    return  np.array(FSI_list)


if __name__ == "__main__":
    import os
    config_file_path = 'configs/face_units/find_face_units_test_mac.json'
    save = True

    np.set_printoptions(precision=3, suppress=True, linewidth=150)

    # load find_face config
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    model = load_model(config)
    # print(model.summary())

    # compute face units
    face_units = found_face_units(model, config)
    print("Shape face_units", np.shape(face_units))

    # save face units
    if save:
        np.save(os.path.join(config['save_path'], config['model']), face_units)

    for f, face_unit in enumerate(face_units):
        num_face_units = len(face_unit[0])
        if num_face_units > 0:
            print(f, "num face units:", num_face_units)
            print(face_unit[0])

