import numpy as np
import tensorflow as tf


def feat_map_filter_processing(pred, ref=None, norm=None, activation='ReLu', filter=None, verbose=False):
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

    # remove reference (compute dynamic)
    if ref is not None:
        ref_pred = np.repeat(np.expand_dims(ref, axis=0), len(pred), axis=0)
        filt_pred = pred - ref_pred

    # compute mean over all maps
    filt_pred = np.mean(filt_pred, axis=3)

    # normalize predictions
    if norm is not None:
        filt_pred = filt_pred / norm

    # get back to initial size
    filt_pred = np.expand_dims(filt_pred, axis=3)

    # apply activation
    if activation == 'ReLu':
        filt_pred[filt_pred < 0] = 0

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
