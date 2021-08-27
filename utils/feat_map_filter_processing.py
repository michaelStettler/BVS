import numpy as np
import tensorflow as tf


def feat_map_filter_processing(pred, ref=None, norm=None, activation='ReLu', filter=None, threshold=None, verbose=False):
    """
    Post processing of the feature maps

    :param pred:
    :param ref:
    :param norm:
    :param activation:
    :param filter:
    :param verbose:
    :return:
    """
    if len(np.shape(pred)) != 4:
        raise ValueError("Input array is not supported! Supported: 4D tensors, "
                         "received shape: {}".format(len(np.shape(pred))))
    # todo set for other array size?
    filt_pred = pred

    # remove reference (compute dynamic)
    if ref is not None:
        ref_pred = np.repeat(np.expand_dims(ref, axis=0), len(pred), axis=0)
        filt_pred = pred - ref_pred

    # normalize predictions
    if norm is not None:
        filt_pred = filt_pred / norm

    # # get back to initial size
    # filt_pred = np.expand_dims(filt_pred, axis=3)
    # print("shape filt_pred", np.shape(filt_pred))

    # apply activation
    if activation == 'ReLu':
        filt_pred[filt_pred < 0] = 0

    # apply threshold
    if threshold is not None:
        filt_pred[filt_pred < threshold] = 0

    if verbose:
        print("[FM_Filt_Proc] Shape filt_pred", np.shape(filt_pred))
        print("[FM_Filt_Proc] Pre-filter: min max filt_pred", np.amin(filt_pred), np.amax(filt_pred))

    # apply filter
    if filter is not None:
        if filter == 'spatial_mean':
            input = tf.convert_to_tensor(filt_pred, dtype=tf.float32)
            kernel = np.ones((3, 3)) / 9
            kernel = tf.convert_to_tensor(np.expand_dims(kernel, axis=(2, 3)),
                                          dtype=tf.float32)  # build 4D kernel with input and output size of 1
            filt_pred = tf.nn.convolution(input, kernel, strides=1, padding='SAME').numpy()
        else:
            raise ValueError("filter : '{}' is not implemented!".format(filter))

    if verbose:
        print("[FM_Filt_Proc] Post-filter: min max filt_pred", np.amin(filt_pred), np.amax(filt_pred))

    return filt_pred


def get_feat_map_filt_preds(preds, config, verbose=False):
    """

    :param preds:
    :param config:
    :param verbose:
    :return:
    """

    # retrive config parameters
    ref_type = None
    if config.get('feat_map_ref_type') is not None:
        ref_type = config['feat_map_ref_type']
    filt_norm = None
    if config.get('feat_map_filt_norm') is not None:
        filt_norm = config['feat_map_filt_norm']
    filt_activation = None
    if config.get('feat_map_filt_activation') is not None:
        filt_activation = config['feat_map_filt_activation']
    filter_type = None
    if config.get('feat_map_filter_type') is not None:
        filter_type = config['feat_map_filter_type']
    threshold = None
    if config.get('feat_map_threshold') is not None:
        threshold = config['feat_map_threshold']

    # declare variables
    filt_preds = []

    print("[Feat. Filtering] Post Processing parametersa - ref_type:{}, norm:{}, activation:{}, filter:{}, threshold:{}".
          format(ref_type, filt_norm, filt_activation, filter_type, threshold))

            # loop over each feture map idx to retain only the one of interest
    for i in range(np.shape(preds)[-1]):
        pred = np.expand_dims(preds[..., i], axis=3)

        if ref_type is None:
            ref = None
        elif ref_type == "self0":
            ref = pred[0]
        else:
            raise ValueError("ref_type: {} is not yet implemented!".format(ref_type))

        filt_pred = feat_map_filter_processing(pred,
                                               ref=ref,
                                               norm=filt_norm,
                                               activation=filt_activation,
                                               filter=filter_type,
                                               threshold=threshold,
                                               verbose=verbose)
        filt_preds.append(filt_pred)

    # build prediction to match the common (n_images, size, size, n_ft) dimensions
    filt_preds = np.array(filt_preds)
    filt_preds = np.squeeze(filt_preds)
    filt_preds = np.moveaxis(filt_preds, 0, -1)
    return filt_preds
