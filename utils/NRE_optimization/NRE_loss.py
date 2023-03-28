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


def prob_neutral(x, rho):
    d = tf.reduce_sum(tf.norm(x, axis=2), axis=1)
    return 1 - (1 / (1 + tf.exp(-(d - rho))))


def prob_expression(proj, p_neut):
    proj = proj - tf.math.reduce_max(proj, axis=1, keepdims=True) # Eliminate overflow and underflow in exponential
    exp = tf.exp(proj)
    # print('exp:', exp)
    sum_exp = tf.reduce_sum(exp, axis=1)
    soft = exp / tf.expand_dims(sum_exp, axis=1)
    return (1 - tf.expand_dims(p_neut, axis=1)) * soft


def compute_loss_with_ref2(x: tf.Tensor, proj: tf.Tensor, y: tf.Tensor, rho: float, alpha_ref=6, n_cat=7):
    """

    :param x: (n_img, n_ft_maps, n_dim)
    :param proj: (n_img, n_ft_maps, n_cat)
    :param y: (n_img, )
    :param radius: float
    :return:
    """

    proj = tf.reduce_sum(proj, axis=1)
    proj = proj[:, 1:]

    # p_neut = prob_neutral(x, rho)
    # weighted version

    ### I think that increasing p_neut and then renormalizing to a probability distribution introduces some unintended behavior
    # p_neut = prob_neutral(x, rho) * alpha_ref
    # p_expr = prob_expression(proj, p_neut)
    # prob = tf.concat((tf.expand_dims(p_neut, axis=1), p_expr), axis=-1)


    # by hand
    p_neut = prob_neutral(x, rho)
    p_expr = prob_expression(proj, p_neut)
    prob = tf.concat((tf.expand_dims(p_neut, axis=1), p_expr), axis=-1)

    alpha = [alpha_ref] + (prob.shape[1] - 1) * [1]
    alpha = tf.expand_dims(tf.convert_to_tensor(alpha, dtype='float'), 0)
    one_hot = tf.one_hot(y, depth=7)
    # print(tf.math.reduce_min(p_expr, axis=1))
    logs = tf.math.log(prob + 1e-10)
    product = logs * one_hot
    product = alpha * product
    entropy = tf.reduce_sum(product)
    if tf.math.is_nan(entropy):
        raise ValueError('Entropy got killed')
    loss = - entropy


    # # use dicrete labels but cannot have sample_weight
    # scce = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss = scce(y, prob)

    # # use one hot since we want the sample weight
    # sw = tf.ones_like(prob)
    # print("shape sw", sw.shape)
    #
    # # create indices for the first column of each matrix
    # indices = tf.stack([tf.range(sw.shape[0]), tf.zeros((sw.shape[0]), dtype=tf.int32)], axis=1)
    # print("shape indices", indices.shape)
    #
    # shifts_updates = tf.ones(sw.shape[0]) * alpha_ref
    # print("shape shifts", shifts_updates.shape)
    #
    # # update weights
    # sw = tf.tensor_scatter_nd_update(sw, indices, shifts_updates)
    # print("shape sw updated", sw.shape)
    #
    # cce = tf.keras.losses.CategoricalCrossentropy()
    # print("shape y", y.shape)
    # print("shape prob", prob.shape)
    # y_hot = tf.one_hot(y, n_cat)
    # print("shape y_hot", y_hot.shape)
    # loss = cce(y_hot, prob, sample_weight=sw)
    # print("shape loss", loss.shape)

    return loss