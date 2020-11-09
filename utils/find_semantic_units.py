import json
import numpy as np
import tensorflow as tf
import cv2

from utils.load_data import load_data
from utils.load_model import load_model


def find_semantic_units(model, data, label):
    input_size = np.shape(data)[1:]
    n_class = np.shape(label)[-1]
    print("[IoU] num classes", n_class)

    # collect all predictions for each layer (discard input)
    # for layer in model.layers[1:]:
    for layer in model.layers[1:2]:
        print("[IoU] layer name:", layer.name)

        # cut model
        model = tf.keras.Model(inputs=model.input, outputs=layer.output)

        # predict face and non_face outputs
        preds = model.predict(data)
        n_unit = np.shape(preds)[-1]
        print("[IoU] shape preds", np.shape(preds))
        print("[IoU] num units", n_unit)

        # get top quantile level Tk
        tk = _compute_quantile(preds)

        # get scaled masked
        sk = _scale_activations(preds, input_size)

        # get thesholded activation
        mk = np.zeros(np.shape(sk), dtype=np.int8)
        mk[sk >= tk] = 1

        # get IoU score for each k, c pairs
        IoU = np.zeros((n_class, n_unit))
        for c in range(n_class):
            for k in range(n_unit):
                l = label[:, :, :, c]
                m = mk[:, :, :, k]
                inter = np.multiply(m, l)
                sum_inter = np.count_nonzero(inter)
                sum_union = np.count_nonzero(m) + np.count_nonzero(l)
                IoU[c, k] = sum_inter / sum_union
                # print(c, k, sum_inter, sum_union, sum_inter / sum_union)

    print("[IoU] finish computing IoU", np.shape(IoU))
    print()


def _compute_quantile(preds, level=0.005):
    """
    Compute top level quantile over the activation distribution ak such that P(ak > Tk) = 0.005

    :param preds:
    :return:
    """
    # move pred's last axis to first axis to have the number of unit in the first one
    ak = np.moveaxis(preds, -1, 0)
    # flatten activation
    ak = np.reshape(ak, (np.shape(ak)[0], -1))
    # sort activations
    ak = np.sort(ak, axis=1)
    # get quantile at 0.005
    # https://www.statisticshowto.com/quantile-definition-find-easy-steps/
    q = (np.shape(ak)[1] + 1) * level
    # get quantile values
    tk = ak[:, -int(q)]

    return tk


def _scale_activations(preds, output_size):
    """
    reshape each unit to the input_size dim (without the depth channel)

    :param preds:
    :param output_size:
    :return:
    """
    dim = (output_size[0], output_size[1])
    #  move last axis to second position to have (image, units, size, size)
    preds = np.moveaxis(preds, -1, 1)
    sk = [[cv2.resize(a, dim, interpolation=cv2.INTER_LINEAR) for a in filter] for filter in preds]
    sk = np.moveaxis(sk, 1, -1)

    return np.array(sk)


if __name__ == "__main__":
    np.random.seed(0)
    import os
    config_file_path = 'configs/semantic_units/find_semantic_units_test_mac.json'
    save = True

    np.set_printoptions(precision=3, suppress=True, linewidth=150)

    # load find_face config
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    #  load model
    model = load_model(config)
    # print(model.summary())

    # load data
    data = np.random.rand(5, 224, 224, 3)  # todo load data
    print("[Loading] shape data", np.shape(data))
    label = np.zeros((5, 224, 224, 2), dtype=np.int8)
    label[0, :, 0, 0] = 1
    label[1, :, :1, 0] = 1
    label[2, 0, :, 0] = 1
    label[3, :1, :, 0] = 1
    label[4, :, -2:, 1] = 1
    print("[Loading] shape label", np.shape(label))
    print("[loading] finish loading data")
    print()

    # compute face units
    find_semantic_units(model, data, label)
