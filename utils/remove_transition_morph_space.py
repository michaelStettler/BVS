import numpy as np


def remove_transition_frames(data, cut=15):
    labels = data[1]
    images = data[0]
    c0_labels = labels[labels == 0]
    c0 = images[labels == 0]
    c1_labels = labels[labels == 1]
    c1 = images[labels == 1]
    c2_labels = labels[labels == 2]
    c2 = images[labels == 2]
    c3_labels = labels[labels == 3]
    c3 = images[labels == 3]
    c4_labels = labels[labels == 4]
    c4 = images[labels == 4]
    print("shape c0", np.shape(c0))
    print("shape c1", np.shape(c1))
    print("shape c2", np.shape(c2))
    print("shape c3", np.shape(c3))
    print("shape c4", np.shape(c4))

    c1_labels = c1_labels[cut:]
    c1_labels = c1_labels[:-cut]
    c1 = c1[cut:]
    c1 = c1[:-cut]
    c2_labels = c2_labels[cut:]
    c2_labels = c2_labels[:-cut]
    c2 = c2[cut:]
    c2 = c2[:-cut]
    c3_labels = c3_labels[9:]
    c3_labels = c3_labels[:-24]
    c3 = c3[9:]
    c3 = c3[:-24]
    c4_labels = c4_labels[cut:]
    c4_labels = c4_labels[:-cut]
    c4 = c4[cut:]
    c4 = c4[:-cut]

    labels = np.concatenate((c0_labels, c1_labels, c2_labels, c3_labels, c4_labels))
    images = np.concatenate((c0, c1, c2, c3, c4))
    print("shape labels", np.shape(labels))
    print("shape images", np.shape(images))

    return [images, labels]