import numpy as np
from tqdm import tqdm

from utils.Metrics.accuracy import compute_accuracy
from utils.NormReference.tuning_vectors import compute_projections


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


def infer_ref_vector(lmk_pos, labels, avatar_labels, n_avatar, avatar_of_int, expr_to_infer, tuning_vectors):
    """
    create one ref vector for each identity by inferring it from the tun_vectors

    :param lmk_pos:
    :param labels:
    :param avatar_labels:
    :param n_avatar:
    :return:
    """
    inf_ref_vectors = []

    for a in range(n_avatar):
        # filter by avatar
        avatar_train_data = lmk_pos[avatar_labels == a]
        avatar_train_label = labels[avatar_labels == a]

        # keep normal ref from the avatar of interest
        if a == avatar_of_int:
            # filter by neutral ref
            avatar_ref_train_data = avatar_train_data[avatar_train_label == 0]

            # add only first (neutral) image from the avatar
            ref_vector = avatar_ref_train_data[0]

        # infer reference from tuning vectors
        else:
            # filter by expression of interest
            avatar_ref_train_data = avatar_train_data[avatar_train_label == expr_to_infer]

            # get first image from the expression of interest
            ref_vector = avatar_ref_train_data[0]

            # infer ref vector from the tuning vectors
            ref_vector -= tuning_vectors[expr_to_infer]

        # append ref vector
        inf_ref_vectors.append(ref_vector)

    return np.array(inf_ref_vectors)


def optimize_inferred_ref(lmk_pos, labels, avatar_labels, n_avatar, avatar_of_int, expr_to_infer,
                           ref_vectors, tun_vectors):

    optimized_ref_vect = np.copy(ref_vectors)

    for a in range(n_avatar):
        print("avatar:", a)
        if a != avatar_of_int:
            # filter by avatar
            av_data = lmk_pos[avatar_labels == a]
            av_labels = labels[avatar_labels == a]
            av_avatar_labels = avatar_labels[avatar_labels == a]

            # filter by expression of interest
            avatar_ref_train_data = av_data[av_labels == expr_to_infer]

            # loop over all images of this avatar and expression
            new_accuracy = None
            new_ref_vectors = ref_vectors
            for i in tqdm(range(len(avatar_ref_train_data))):
                lmk_vector = avatar_ref_train_data[i]

                # infer the ref vector from the lmk vector expression
                inferred_ref = lmk_vector - tun_vectors[expr_to_infer]

                # update ref vectors with new inferred vector
                new_ref_vectors[a] = inferred_ref

                # compute new accuracy on the avatar data
                projections_preds = compute_projections(av_data, av_avatar_labels, new_ref_vectors,
                                                        tun_vectors,
                                                        neutral_threshold=0,
                                                        verbose=False)

                # compute accuracy
                accuracy = compute_accuracy(projections_preds, av_labels)

                if new_accuracy is None:
                    print(f"idx: {i}, new_accuracy: {accuracy}")
                    new_accuracy = accuracy
                elif accuracy > new_accuracy:
                    # update ref_vectors
                    print(f"idx: {i}, new_accuracy: {accuracy}")
                    optimized_ref_vect[a] = inferred_ref
                    new_accuracy = accuracy

    return optimized_ref_vect

