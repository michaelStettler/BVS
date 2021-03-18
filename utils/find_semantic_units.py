import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm


"""
take only labels that has at least 10 images 
"""


def find_semantic_units(model, data, label):
    input_size = np.shape(data)[1:]
    n_class = np.shape(label)[-1]
    print("[IoU] num classes", n_class)

    # collect all predictions for each layer (discard input)
    for layer in tqdm(model.layers[1:]):
        print("[IoU] layer name:", layer.name)

        # stop if the layers flatten tha array since the resize won't work
        # todo take care of flatten units for reconstruction?
        if 'flatten' in layer.name:
            break

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

        # get number of unit per category
        IoU[IoU < 0.04] = 0
        n_units = np.count_nonzero(IoU, axis=1)
        print("n_units")
        print(n_units)
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

