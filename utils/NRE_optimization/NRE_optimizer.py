import os

import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils.NRE_optimization.NRE_vectors import compute_tun_vectors
from utils.NRE_optimization.NRE_loss import compute_loss_with_ref
from utils.NRE_optimization.NRE_loss import compute_loss_without_ref
from utils.NRE_optimization.NRE_loss import compute_loss_with_ref2
from plots_utils.plot_NRE_optimizers import plot_space


def batch(x, y, n=32):
    l = x.shape[0]
    for ndx in range(0, l, n):
        yield x[ndx:min(ndx + n, l)], y[ndx:min(ndx + n, l)]


def compute_NRE_preds(projections, radius, use_ref=False):
    # everything that is under the radius is considered as category ref
    if use_ref:
        # get only the max proj to see if they are within the circle
        max_proj = np.amax(projections, axis=2)
        # consider only the absolute values as we want to be in the circle
        ref_proj = np.abs(max_proj)
        # set to zero all values within the radius (they don't count)
        ref_proj[ref_proj < radius] = 0
        # compute sum per entry (need that all pts are within the radius)
        ref_proj = np.sum(ref_proj, axis=1)
        # if the sum is zero, that means we are everywhere within the radius, thus it is a ref point
        ref_proj[ref_proj > 0] = -1
        ref_proj[ref_proj == 0] = float("inf")
        ref_proj[ref_proj < 1] = 0
        # set ref values
        projections[..., 0] = np.repeat(np.expand_dims(ref_proj, axis=1), projections.shape[1], axis=1)

    # add projections per feature maps
    predictions = np.sum(projections, axis=1)

    return predictions



def compute_projections(x, tun_vectors) -> np.array:
    """
    :param x: (n_img, n_feat_map, n_dim)
    :param tun_vectors: (n_cat, n_feat_map, n_dim)
    :param nu:
    :return:
    """
    # case where there's no entry in x
    if x.ndim == 0:
        return tf.zeros((0, x.shape[0], x.shape[1]), dtype=tf.float32)

    # batch per ft_map (meaning putting ft_map dim in first column)
    x = tf.experimental.numpy.moveaxis(x, 0, 1)
    tun_vect = tf.experimental.numpy.moveaxis(tun_vectors, 0, -1)
    projections = tf.matmul(x, tun_vect)  # does not take care of norm_t == 0
    # put back ft_map dim in the middle -> (n_img, n_feat_map, n_dim)
    projections = tf.experimental.numpy.moveaxis(projections, 1, 0)

    return projections


def compute_distance(x: tf.Tensor, radius: tf.Tensor):
    """
    :param x: (n_img, n_feat_map, n_dim)
    :param radius:
    :return
    """
    if x.ndim == 2:
        return tf.zeros((0, x.shape[0], x.shape[1]), dtype=tf.float32)
    # return tf.exp(- tf.norm(x, axis=2) / radius)
    return 1 / (1 + tf.exp(-(tf.norm(x, axis=2) - radius)))


# @tf.function  # create a graph (non-eager mode!)
# def optimize_NRE(x, y, n_cat, use_ref=True, batch_size=32, n_ref=1, init_ref=None, lr=0.01, n_epochs=20,
#                  alpha_ref=1, do_plot=False, plot_alpha=1, plot_name="NRE_optimizer", min_plot_axis=15,
#                  max_plot_axis=15, early_stopping=False):
#     """
#     :param x: (n_pts, n_feature_maps, n_dim)
#     :param y:
#     :param neutral:
#     :return:
#     """
#
#     n_dim = tf.shape(x)[-1]
#     n_feat_maps = tf.shape(x)[1]
#
#     # initialize trainable parameters
#     shifts = tf.zeros((n_ref, n_feat_maps, n_dim), dtype=tf.float32, name="shifts")
#     if init_ref is not None:
#         shifts = tf.identity(init_ref, name="shifts")
#     print("shape shifts", shifts.shape)
#     # t_shifts = tf.zeros((n_feat_maps, n_dim), dtype=tf.float32, name="t_shifts")
#     # print("shape t_shifts", t_shifts.shape)
#     radius = tf.ones(1, dtype=tf.float32, name="radius")
#     print("shape radius", radius.shape)
#     best_acc = 0
#
#     best_ref = None
#     best_radius = None
#     best_tuning = None
#
#     # declare sequence parameters
#     if do_plot:
#         path = ""
#         video_name = f"{plot_name}.mp4"
#         n_rows = int(np.sqrt(n_feat_maps))
#         n_columns = np.ceil(n_feat_maps / n_rows).astype(int)
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video = cv2.VideoWriter(os.path.join(path, video_name), fourcc, 30, (n_columns * 400, n_rows * 400))
#
#     for epoch in range(n_epochs):
#         it = 0
#         predictions = []
#         for x_batch, y_batch in batch(x, y, n=batch_size):
#             batch_shifts = tf.zeros((x_batch.shape[0], n_feat_maps, n_dim), dtype=tf.float32, name="batch_shifts")
#             # batch_radius = tf.zeros((x_batch.shape[0], n_feat_maps), dtype=tf.float32, name="batch_radius")
#             #print("shape x_batch", x_batch.shape, "shape y_batch", y_batch.shape)
#             loss = 0
#
#             with tf.GradientTape() as tape:
#                 tape.watch(shifts)
#                 # tape.watch(t_shifts)
#                 tape.watch(radius)
#
#                 # set batch_shifts and batch_radius to match the category according to their label (y[, 1])
#                 for r in range(n_ref):
#                     # print("epoch:", epoch, "batch it:", it, "n_ref", r)
#                     # filter data per ref
#                     ref_mask = tf.equal(y_batch[:, 1], r)
#
#                     # get indices of mask
#                     indices = tf.where(ref_mask)
#
#                     # construct updates values
#                     rep_shifts = tf.repeat(tf.expand_dims(shifts[r], axis=0), x_batch.shape[0],
#                                            axis=0, name="rep_shifts")
#                     shifts_updates = tf.gather(rep_shifts, tf.squeeze(indices))
#
#                     # assign value like: batch_shifts[indices] = shifts[indices]
#                     batch_shifts = tf.tensor_scatter_nd_update(batch_shifts, indices, shifts_updates)
#
#                 # subtract  shifts to x
#                 x_shifted = tf.subtract(x_batch, batch_shifts, name="x_shifted")
#                 # print("shape x_shifted", x_shifted.shape)
#                 # print(x_shifted)
#
#                 # get tun vectors
#                 tun_vectors = compute_tun_vectors(x_shifted, y_batch[:, 0], n_cat, use_ref=use_ref)
#
#                 # if epoch == 0 and it == 0:
#                 #     tun_vectors = compute_tun_vectors(x_shifted, y_batch[:, 0], n_cat, use_ref=use_ref)
#                 # tun_vectors += t_shifts
#
#                 # print("tun_vectors", tun_vectors.shape)
#                 # print(tun_vectors)
#
#                 # # get projections
#                 projections = compute_projections(x_shifted, tun_vectors)
#                 # print("projections", projections.shape)
#                 # print(projections)
#
#                 # compute preds
#                 batch_preds = compute_NRE_preds(projections.numpy(), radius.numpy(), use_ref=use_ref)
#
#                 if use_ref:
#                     # sig_distance = compute_distance(x_shifted, batch_radius)
#
#                     # compute loss
#                     # loss += compute_loss_with_ref(projections, y_batch[:, 0], sig_distance, alpha_ref=alpha_ref)
#                     loss += compute_loss_with_ref2(x_shifted, projections, y_batch[:, 0], radius, alpha_ref=alpha_ref)
#
#                 else:
#                     # compute loss
#                     loss += compute_loss_without_ref(projections, y_batch[:, 0])
#
#             # compute accuracy
#             y_pred = np.argmax(batch_preds, axis=1)  # sum vectors over all feature space
#             predictions.append(y_pred)
#             acc = accuracy_score(y_batch[:, 0], y_pred)
#
#             # print(f"{epoch} loss {loss}, radius[0]: {radius[0]}", end='\r')
#             print(f"it: {it}, loss={loss:.4f}, train_acc={acc:.3f}", end='\r')
#
#             # compute gradients
#             grad_shifts, grad_radius = tape.gradient(loss, [shifts, radius])
#             # grad_shifts, grad_radius, grad_t_shifts = tape.gradient(loss, [shifts, radius, t_shifts])
#             # print("grad shifts", grad_shifts.shape)
#
#             # update parameters
#             shifts = shifts - lr * grad_shifts
#             radius = radius - lr * grad_radius
#             # t_shifts = t_shifts - lr * grad_t_shifts
#             # print(f"{epoch} shifts {shifts}")
#             # print()
#
#             # increase iteration
#             it += 1
#
#             # compute accuracy
#             y_pred = np.argmax(batch_preds, axis=1)  # sum vectors over all feature space
#             acc = accuracy_score(y_batch[:, 0], y_pred)
#             print(f"it: {it}, loss={loss:.4f}, train_acc={acc:.3f}", end='\r')
#
#         if do_plot:
#             tun_vect = tun_vectors.numpy()
#             # img = plot_space(x.numpy(), y.numpy(), n_cat, shifts=shifts.numpy(), tun_vectors=tun_vect)
#             img = plot_space(x.numpy(), y.numpy(), n_cat,
#                              shifts=shifts.numpy(),
#                              tun_vectors=tun_vect,
#                              alpha=plot_alpha,
#                              min_axis=min_plot_axis,
#                              max_axis=max_plot_axis)
#
#             # write image
#             video.write(img)
#
#         # compute epoch accuracy
#         print(f"it: {it}, loss={loss:.4f}, train_acc={acc:.3f}")  # simply to re-print because of the EOL
#         epoch_acc, predictions = estimate_NRE(x, y,
#                                               {'references': shifts, 'radius': radius, 'tun_vectors': tun_vectors},
#                                               n_ref=n_ref,
#                                               batch_size=batch_size,
#                                               use_ref=use_ref,
#                                               verbose=False)
#         print(f"epoch: {epoch}, epoch_acc={epoch_acc}")
#
#         # save best
#         if epoch_acc > best_acc:
#             print("save parameters")
#             best_acc = epoch_acc
#             best_ref = shifts
#             best_radius = radius
#             best_tuning = tun_vectors
#
#         # apply early stopping
#         if early_stopping and epoch_acc < best_acc - 0.01:
#             print()
#             print(f"Early stopping at {epoch}!")
#             print("diff:", best_acc - epoch_acc)
#             break
#
#     if do_plot:
#         cv2.destroyAllWindows()
#         video.release()
#
#     # print last one to keep in the log
#     print(f"best_acc={best_acc}")
#     print("best ref[0]", best_ref[0])
#     print("best_radius", best_radius)
#     # print("predictions")
#     # print(predictions)
#     print("y_pred", np.shape(y_pred))
#     # print(y_pred)
#
#     return predictions, {'references': best_ref, 'radius': best_radius, 'tun_vectors': best_tuning}


def estimate_NRE(x, y, params, use_ref=True, batch_size=32, n_ref=1, verbose=True):

    refs = params['references']
    radius = params['radius']
    tun_vectors = params['tun_vectors']

    n_dim = tf.shape(x)[-1]
    n_feat_maps = tf.shape(x)[1]

    predictions = np.array([])
    for x_batch, y_batch in batch(x, y, n=batch_size):
        batch_shifts = tf.zeros((x_batch.shape[0], n_feat_maps, n_dim), dtype=tf.float32, name="batch_shifts")

        for r in range(n_ref):
            # print("epoch:", epoch, "batch it:", it, "n_ref", r)
            # filter data per ref
            ref_mask = tf.equal(y_batch[:, 1], r)

            # get indices of mask
            indices = tf.where(ref_mask)

            # construct updates values
            rep_shifts = tf.repeat(tf.expand_dims(refs[r], axis=0), x_batch.shape[0], axis=0, name="rep_shifts")
            shifts_updates = tf.gather(rep_shifts, tf.squeeze(indices))

            # assign value like: batch_shifts[indices] = shifts[indices]
            batch_shifts = tf.tensor_scatter_nd_update(batch_shifts, indices, shifts_updates)

        # subtract  shifts to x
        x_shifted = tf.subtract(x_batch, batch_shifts, name="x_shifted")

        # # get projections
        projections = compute_projections(x_shifted, tun_vectors)

        # compute preds
        batch_preds = compute_NRE_preds(projections.numpy(), radius.numpy(), use_ref=use_ref)

        # get predictions
        y_pred = np.argmax(batch_preds, axis=1)
        if len(predictions) == 0:
            predictions = y_pred
        else:
            predictions = np.concatenate((predictions, y_pred))

    # compute accuracy
    acc = accuracy_score(predictions, y[:, 0])
    if verbose:
        print(f"accuracy={acc}")

        print(confusion_matrix(y[:, 0], predictions))
        print(classification_report(y[:, 0], predictions))

    return acc, predictions


def fit_NRE(x, y, n_cat, x_test=None, y_test=None, use_ref=True, batch_size=32, n_ref=1, init_ref=None, lr=0.01, n_epochs=20,
                 alpha_ref=1, plot_alpha=1, plot_name="NRE_optimizer", min_plot_axis=15,
                 max_plot_axis=15, early_stopping=False, lr_decay=[75, 200, 300]):
    """
    :param x: (n_pts, n_feature_maps, n_dim)
    :param y:
    :param neutral:
    :return:
    """

    is_test_set = False
    if x_test is not None and y_test is not None:
        is_test_set = True

    n_dim = tf.shape(x)[-1]
    n_feat_maps = tf.shape(x)[1]

    # initialize trainable parameters
    shifts = tf.zeros((n_ref, n_feat_maps, n_dim), dtype=tf.float32, name="shifts")
    if init_ref is not None:
        shifts = tf.identity(init_ref, name="shifts")
    print("shape shifts", shifts.shape)
    radius = tf.ones(1, dtype=tf.float32, name="radius")
    print("shape radius", radius.shape)
    best_acc = 0

    best_ref = None
    best_radius = None
    best_tuning = None
    train_accuracies = []
    test_accuracies = []

    for epoch in range(n_epochs):
        it = 0
        predictions = []
        for x_batch, y_batch in batch(x, y, n=batch_size):
            batch_shifts = tf.zeros((x_batch.shape[0], n_feat_maps, n_dim), dtype=tf.float32, name="batch_shifts")
            loss = 0

            with tf.GradientTape() as tape:
                tape.watch(shifts)
                tape.watch(radius)

                # set batch_shifts and batch_radius to match the category according to their label (y[, 1])
                for r in range(n_ref):
                    # filter data per ref
                    ref_mask = tf.equal(y_batch[:, 1], r)

                    # get indices of mask
                    indices = tf.where(ref_mask)

                    # construct updates values
                    rep_shifts = tf.repeat(tf.expand_dims(shifts[r], axis=0), x_batch.shape[0],
                                           axis=0, name="rep_shifts")
                    shifts_updates = tf.gather(rep_shifts, tf.squeeze(indices))

                    # assign value like: batch_shifts[indices] = shifts[indices]
                    batch_shifts = tf.tensor_scatter_nd_update(batch_shifts, indices, shifts_updates)

                # subtract  shifts to x
                x_shifted = tf.subtract(x_batch, batch_shifts, name="x_shifted")


                # get tun vectors
                tun_vectors = compute_tun_vectors(x_shifted, y_batch[:, 0], n_cat, use_ref=use_ref)
                # if epoch == 0 and it == 0:
                #     tun_vectors = compute_tun_vectors(x_shifted, y_batch[:, 0], n_cat, use_ref=use_ref)
                # tun_vectors += t_shifts

                # get projections
                projections = compute_projections(x_shifted, tun_vectors)

                # compute preds
                batch_preds = compute_NRE_preds(projections.numpy(), radius.numpy(), use_ref=use_ref)

                if use_ref:
                    batch_loss = compute_loss_with_ref2(x_shifted, projections, y_batch[:, 0], radius, alpha_ref=alpha_ref)
                else:
                    # compute loss
                    batch_loss = compute_loss_without_ref(projections, y_batch[:, 0])
                # Stop optimization if the loss breaks down
                if batch_loss == None:
                    break
                loss += batch_loss

            # compute accuracy
            y_pred = np.argmax(batch_preds, axis=1)  # sum vectors over all feature space
            predictions.append(y_pred)
            acc = accuracy_score(y_batch[:, 0], y_pred)
            print(f"it: {it}, loss={loss:.4f}, train_acc={acc:.3f}", end='\r')

            # compute gradients
            grad_shifts, grad_radius = tape.gradient(loss, [shifts, radius])

            # check whether grads are nan
            nan = tf.experimental.numpy.ravel(tf.math.is_nan(grad_shifts)).numpy()
            nan = list(nan)
            if sum(nan) > 0:
                print('grad shifts are nan!')
                print('Number of nan values in tensor:', sum(nan))

                # remove nans from the gradient
                nan_indices = tf.where(tf.math.is_nan(grad_shifts))
                grad_shifts = tf.tensor_scatter_nd_update(
                    grad_shifts,
                    nan_indices,
                    tf.zeros((tf.shape(nan_indices)[0]))
                )

            nan = tf.experimental.numpy.ravel(tf.math.is_nan(grad_radius)).numpy()
            nan = list(nan)
            if sum(nan) > 0:
                print('grad radius is nan!')
                print('Number of nan values in tensor:', sum(nan))

                # remove nans from the gradient
                nan_indices = tf.where(tf.math.is_nan(grad_radius))
                grad_radius = tf.tensor_scatter_nd_update(
                    grad_radius,
                    nan_indices,
                    tf.zeros((tf.shape(nan_indices)[0]))
                )



            # update parameters
            shifts = shifts - lr * grad_shifts
            radius = radius - lr * grad_radius

            # increase iteration
            it += 1

            # compute accuracy
            y_pred = np.argmax(batch_preds, axis=1)  # sum vectors over all feature space
            acc = accuracy_score(y_batch[:, 0], y_pred)
            print(f"it: {it}, loss={loss:.4f}, train_acc={acc:.3f}", end='\r')

        # compute epoch accuracy
        print(f"it: {it}, loss={loss:.4f}, train_acc={acc:.3f}")  # simply to re-print because of the EOL
        epoch_train_acc, predictions = estimate_NRE(x, y,
                                                    {'references': shifts, 'radius': radius, 'tun_vectors': tun_vectors},
                                                    n_ref=n_ref,
                                                    batch_size=batch_size,
                                                    use_ref=use_ref,
                                                    verbose=False)
        train_accuracies.append(epoch_train_acc)

        if is_test_set:
            epoch_test_acc, _ = estimate_NRE(x, y,
                                             {'references': shifts, 'radius': radius,
                                              'tun_vectors': tun_vectors},
                                             n_ref=n_ref,
                                             batch_size=batch_size,
                                             use_ref=use_ref,
                                             verbose=False)
            test_accuracies.append(epoch_test_acc)

        if is_test_set:
            print(f"epoch: {epoch}, train_acc={epoch_train_acc}, test_acc={epoch_test_acc}")
        else:
            print(f"epoch: {epoch}, train_acc={epoch_train_acc}")

        # lower learning rate after
        if epoch in lr_decay:
            print("Lowered lr by order of magnitude")
            lr *= 0.1

        # save best
        if epoch_train_acc > best_acc:
            print("save parameters")
            best_acc = epoch_train_acc
            best_ref = shifts
            best_radius = radius
            best_tuning = tun_vectors

        # apply early stopping
        if early_stopping and epoch_train_acc < best_acc - 0.01:
            print()
            print(f"Early stopping at {epoch}!")
            print("diff:", best_acc - epoch_train_acc)
            break

    if is_test_set:
        metrics = {'best_acc': best_acc, 'train_accuracies': train_accuracies, 'test_accuracies': test_accuracies}
    else:
        metrics = {'best_acc': best_acc, 'train_accuracies': train_accuracies}

    return predictions, {'references': best_ref, 'radius': best_radius, 'tun_vectors': best_tuning}, metrics