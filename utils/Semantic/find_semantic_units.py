import os
import numpy as np
import pickle
import tensorflow as tf
import cv2
from tqdm import tqdm


"""
Methods that implement the paper: 
Network Dissection: Quantifying Interpretability of Deep Visual Representations

from David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba

"""

# todo: take only labels that has at least 10 images


def find_semantic_units(model, data, config, verbose=2, save=False):
    """
    verbose = 1: only progress bar
    berbose = 2: all print

    :param model:
    :param data:
    :param config:
    :param verbose:
    :param save:
    :return:
    """
    label = data[1]
    data = data[0]

    input_size = np.shape(data)[1:]
    n_class = np.shape(label)[-1]

    if verbose > 1:
        print("[IoU] num classes", n_class)

    # silence progress bar if not wanted
    disable_bar = False
    if verbose < 1:
        disable_bar = True

    # run over each layer
    sem_indexes = {}
    # collect all predictions for each layer (discard input)
    for l_idx, layer in enumerate(tqdm(model.layers[1:], disable=disable_bar)):
        layer_index = {"layer_name": layer.name, "layer_idx": l_idx}

        # stop if the layers flatten tha array since the resize won't work
        # todo take care of flatten units for reconstruction?
        if 'flatten' in layer.name:
            break

        # cut model
        model = tf.keras.Model(inputs=model.input, outputs=layer.output)

        # predict face and non_face outputs
        preds = model.predict(data)
        n_unit = np.shape(preds)[-1]

        # get top quantile level Tk
        tk = _compute_quantile(preds)

        # get scaled masked
        sk = _scale_activations(preds, input_size)

        # get thresholded activation
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
                if sum_union == 0:  # -> this also mean that inter = 0
                    IoU[c, k] = 0
                else:
                    IoU[c, k] = sum_inter / sum_union
                # print(c, k, sum_inter, sum_union, sum_inter / sum_union)

        # get number of unit per category
        IoU[IoU < 0.04] = 0
        n_units = np.count_nonzero(IoU, axis=1)

        if verbose > 1:
            print("[IoU] layer {} name {}:".format(l_idx, layer.name))
            print("[IoU] shape preds", np.shape(preds))
            print("[IoU] num units", n_unit)
            print("n_units")
            print(n_units)

        layer_index["IoU"] = IoU
        sem_indexes[layer.name] = layer_index

    # sem_indexes = np.array(sem_indexes)
    if verbose > 1:
        print("[IoU] finished computing IoU")

    if save:
        save_folder = os.path.join("models/saved", config["config_name"])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        with open(os.path.join(save_folder, 'semantic_dictionary.pkl'), 'wb') as f:
            pickle.dump(sem_indexes, f, pickle.HIGHEST_PROTOCOL)

    return sem_indexes


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


def get_IoU_per_category(IoU_dict, cat_ids, sort=True):
    """
    Get the feature map index for the IoU dictionary and the category index of interest
    If sort is true, it returns the index from the highest to the lowest score

    :param IoU_dict: IoU score dictionary
    :param cat_ids: category index
    :param sort:
    :return:
    """
    cat_indexes = {}
    # sort for each categories over each layers the feature map indexes that are activated
    for cat in cat_ids:
        layer_index = {}
        for layer_name in IoU_dict:
            IoU = IoU_dict[layer_name]["IoU"]
            # get the non zero index
            non_zero_idx = np.nonzero(IoU[cat])[0]

            if sort:
                # sort index from biggest to smallest IoU index
                IoU_idx = np.flip(np.argsort(IoU[cat]))
                # retain only the non zero index
                IoU_idx = IoU_idx[:len(non_zero_idx)]
            else:
                IoU_idx = non_zero_idx

            layer_index[layer_name] = {"layer_name": IoU_dict[layer_name]["layer_name"], "indexes": IoU_idx}

        cat_indexes["category_{}".format(cat)] = layer_index

    return cat_indexes


