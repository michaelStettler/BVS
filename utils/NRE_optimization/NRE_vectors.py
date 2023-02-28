import tensorflow as tf


def compute_tun_vectors(x, y, n_cat, use_ref=False, type="average"):
    """
    :param type: selection of the point(s) use to conpute the tuning vectors (average, argmax)
    """
    n_feat_maps = x.shape[1]
    n_dim = x.shape[-1]

    # if no data comes in
    if y.shape[0] == 0:
        # careful because as y is empty, then x shape changes to ndim = 2
        return tf.zeros((n_cat, x.shape[0], n_dim), dtype=tf.float32)

    tun_vectors = []
    # for each expression
    for cat in range(n_cat):
        # construct mask to get slice of tensor x per category (in numpy x[y == cat])
        bool_mask = tf.equal(y, cat)
        x_cat = tf.gather(x, tf.squeeze(tf.where(bool_mask)))

        # only if there's sample of this category in the batch
        if x_cat.shape[0] != 0:
            # set ref tuning vectors to zero is use_ref is set and ref == 0
            if use_ref and cat == 0:
                tun_vectors.append(tf.zeros((n_feat_maps, n_dim), dtype=tf.float32))
                continue  # pass the reference

            # if only one point in the category
            if x_cat.ndim == 2:
                x_cat = tf.expand_dims(x_cat, axis=0)

            # declare tuning vect per category
            # v_cat = tf.zeros((n_feat_maps, n_dim))
            v_cat = []
            for f in range(n_feat_maps):
                # print(f"cat: {cat}, feat_map: {f}, x_cat[:, f]:")
                # print(x_cat[:, f])

                # svd results not consistent between torch and tf
                s, u, vh = tf.linalg.svd(x_cat[:, f], full_matrices=False)
                # print("shape u, s, vh", u.shape, s.shape, vh.shape)
                # print(vh)

                # Orient tuning vectors properly
                vh = tf.transpose(vh)
                if type == "average":
                    direction = tf.reduce_mean(x_cat[:, f], axis=0)
                elif type == "argmax":
                    direction = tf.gather(x_cat[:, f], tf.math.argmax(tf.norm(x_cat[:, f], axis=-1)))
                direction = direction * vh[0]

                # get sign
                sign = tf.math.sign(direction)

                # v_cat.append(vh[0])
                tun_vect = [vh[0, 0] * sign[0], vh[0, 1] * sign[1]]
                v_cat.append(tun_vect)

            v_cat = tf.convert_to_tensor(v_cat, dtype=tf.float32)

            tun_vectors.append(v_cat)
        # no point in the category
        else:
            tun_vectors.append(tf.zeros((n_feat_maps, n_dim), dtype=tf.float32))

    return tf.convert_to_tensor(tun_vectors, dtype=tf.float32, name="tun_vectors")