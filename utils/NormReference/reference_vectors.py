import numpy as np


def built_references_vectors(lmk_ref, ref_pos, identities, positions):
    """
    create a reference vector for each identities (images)

    :param lmk_ref:
    :param ref_pos:
    :param identities:
    :param positions:
    :return:
    """
    lmk_ref = np.array(lmk_ref)
    ref_pos = np.array(ref_pos)

    ref_vectors = []
    for i, id in enumerate(identities):
        ref_vectors.append(lmk_ref[id] - (positions[i] - ref_pos[id]))

    return np.array(ref_vectors)


def learn_ref_vector(lmk_pos, labels, avatar_labels, n_avatar):
    """
    create one ref vector for each identity

    :param lmk_pos:
    :param labels:
    :param avatar_labels:
    :param n_avatar:
    :return:
    """
    ref_vectors = []

    for a in range(n_avatar):
        # filter by avatar
        avatar_train_data = lmk_pos[avatar_labels == a]
        avatar_train_label = labels[avatar_labels == a]

        # filter by neutral ref
        avatar_ref_train_data = avatar_train_data[avatar_train_label == 0]

        # add only first from the avatar
        ref_vectors.append(avatar_ref_train_data[0])

    return np.array(ref_vectors)