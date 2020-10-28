import json
import numpy as np
import tensorflow as tf

from utils.load_data import load_data


def found_face_units(model, config):
    x, y = load_data(config)

    x_face = x[:50]
    x_object = x[50:]
    print("shape x", np.shape(x))
    FSI_list = []
    for layer in model.layers[:2]:
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

            FSI_list.append(FSI)

    print("Shape FSI_list", np.shape(FSI_list))


if __name__ == "__main__":
    config_file_path = 'configs/face_units/find_face_units_test.json'

    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    # -----------------------------------------------------------------
    # limit GPU memory as it appear windows have an issue with this, from:
    # https://forums.developer.nvidia.com/t/could-not-create-cudnn-handle-cudnn-status-alloc-failed/108261/3
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # -----------------------------------------------------------------

    # load find_face config
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    model = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    # print(model.summary())

    # compute face units
    found_face_units(model, config)
