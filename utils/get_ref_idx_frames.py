import numpy as np


def get_ref_idx_frames(data, ref_index=0):
    """
    filter data and return only the frames having the "ref_index" as labels

    :param data:
    :param ref_index:
    :return:
    """
    if len(data[0]) != len(data[1]):
        raise ValueError("dataset and labels does not have the same length!")

    preds = data[0]
    labels = data[1]
    neutral_frames = []

    for i in range(len(data[0])):
        if int(labels[i]) == ref_index:
            neutral_frames.append(preds[i])

    return np.array(neutral_frames)
