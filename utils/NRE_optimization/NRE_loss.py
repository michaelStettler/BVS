import tensorflow as tf


def compute_loss_without_ref(proj: tf.Tensor, y: tf.Tensor):
    """

    :param proj: (n_img, n_ft_maps, n_cat)
    :param y: (n_img, )
    :return:
    """

    # compute sum of all exp proj
    # sum_proj = np.sum(np.exp(proj), axis=1)  # don't think this is the correct way for us (email)
    n_img = proj.shape[0]

    # treat neutral as own category
    loss = 0
    for i in range(n_img):
        enumerator = tf.exp(proj[i, :, int(y[i])])
        denominator = tf.reduce_sum(tf.exp(proj[i]), axis=-1)
        loss += tf.reduce_sum(enumerator / denominator)

    return -loss


def compute_loss_with_ref(proj: tf.Tensor, y: tf.Tensor, distance: tf.Tensor, alpha_ref=1):
    """

    :param proj: (n_img, n_ft_maps, n_cat)
    :param y: (n_img, )
    :param distance: (n_img, n_ft_maps)
    :return:
    """

    # remove first column (ref column)
    proj = proj[..., 1:]

    # compute sum of all exp proj
    n_img = proj.shape[0]

    # treat neutral as own category
    loss = 0
    for i in range(n_img):
        # ref sample
        if int(y[i]) == 0:
            loss += alpha_ref * tf.reduce_sum(distance[i])
        # cat sample
        else:
            enumerator = tf.exp(proj[i, :, int(y[i])-1])
            denominator = tf.reduce_sum(tf.exp(proj[i]), axis=-1)
        #     # loss += tf.reduce_sum((1 - distance[i]) * enumerator / denominator)
            loss += tf.reduce_sum(distance[i] * enumerator / denominator)

    return loss


def prob_neutral(x, radius):
    d =
    return 1 - (1 / 1 + tf.exp(-d))


def compute_loss_with_ref2(x: tf.Tensor, proj: tf.Tensor, y: tf.Tensor, radius: float, alpha_ref=1):
    """

    :param x: (n_img, n_ft_maps, n_dim)
    :param proj: (n_img, n_ft_maps, n_cat)
    :param y: (n_img, )
    :param radius: float
    :return:
    """

    print("shape x", x.shape)
    print("shape proj", proj.shape)
    print("shape radius", radius.shape)
    print()


    proj = tf.reduce_sum(proj, axis=1)

    return 0
