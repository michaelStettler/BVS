import numpy as np

from utils.Semantic.filter_with_semantic_units import get_semantic_pred
from utils.LMK.interactive_labelling import get_lmk_on_image
from utils.RBF_patch_pattern.optimize_sigma import optimize_sigma
from utils.RBF_patch_pattern.optimize_sigma import decrease_sigma
from utils.RBF_patch_pattern.lmk_patches import predict_RBF_patch_pattern_lmk_pos


def construct_pattern(pred, pos, size, ratio=1):
    # get padding
    pad_x = int(size[0] / 2)
    pad_y = int(size[1] / 2)

    # get lmk pos relative to the latent space
    x_pos = np.round(pos[0] / ratio).astype(int)
    y_pos = np.round(pos[1] / ratio).astype(int)

    # construct patch
    x_patch = [x_pos - pad_x, x_pos + pad_x + 1]
    y_patch = [y_pos - pad_y, y_pos + pad_y + 1]

    # return pattern
    return pred[0, y_patch[0]:y_patch[1], x_patch[0]:x_patch[1]]  # image x/y are shifted


def construct_patterns(preds, pos, k_size, ratio=1):
    n_lmks = np.shape(pos)[1]
    n_patterns = np.shape(preds)[0]  # == n_images

    pad_x = int(k_size[0] / 2)
    pad_y = int(k_size[1] / 2)
    patterns = np.zeros((n_patterns, n_lmks, k_size[1], k_size[0], np.shape(preds)[-1]))

    for i in range(n_patterns):
        for j in range(n_lmks):
            x_pos = np.round(pos[i, j, 0] / ratio).astype(int)
            y_pos = np.round(pos[i, j, 1] / ratio).astype(int)
            x_patch = [x_pos - pad_x, x_pos + pad_x + 1]
            y_patch = [y_pos - pad_y, y_pos + pad_y + 1]
            patterns[i, j] = preds[i, y_patch[0]:y_patch[1], x_patch[0]:x_patch[1]]  # image x/y are shifted

    return patterns


def label_and_construct_patterns(img, pred, im_ratio=1, k_size=(7, 7), pre_processing='VGG19', fig_title=''):
    # label image
    lmk_pos = get_lmk_on_image(img, im_ratio=im_ratio, pre_processing=pre_processing, fig_title=fig_title)

    if lmk_pos is not None:
        # construct patterns
        pattern = construct_pattern(pred, lmk_pos, k_size, ratio=224 / 56)
        pattern = np.expand_dims(pattern, axis=0)  # add a dimension to mimic 1 landmark dimension
    else:
        print("no landmark positions!")
        pattern = None

    return lmk_pos, pattern


def construct_RBF_patterns(images, v4_model, lmk_type, config, lr_rate=100, init_sigma=100, im_ratio=1, k_size=(7, 7),
                           use_only_last=False, loaded_patterns=None, loaded_sigma=None, train_idx=None, lmk_name=''):
    patterns = []
    sigma = init_sigma
    do_force_label = False

    if loaded_patterns is not None:
        for p in loaded_patterns:
            patterns.append(p)

    if loaded_sigma is not None:
        sigma = loaded_sigma

    label_img_idx = []
    lmk_idx = 0

    if train_idx is not None:
        images = images[train_idx]
        print("shape images", np.shape(images))
        do_force_label = True

    for i, img in enumerate(images):
        print("image: ", i, end='')
        # transform image to latent space
        lat_pred = get_semantic_pred(v4_model, img, lmk_type, config)

        fig_title = "image:{} {}".format(i, lmk_name)

        # label first image if no patterns
        if len(patterns) == 0:
            print(" - need labeling")
            # label image
            lmk_pos, new_pattern = label_and_construct_patterns(img, lat_pred,
                                                                im_ratio=im_ratio,
                                                                k_size=k_size,
                                                                fig_title=fig_title)

            if lmk_pos is not None:
                # save labelled image idx
                patterns.append(new_pattern)
                label_img_idx.append(i)

                # optimize sigma
                sigma = optimize_sigma(images, np.array(patterns), sigma, label_img_idx, init_sigma, lr_rate, is_first=True)
                print("new sigma", sigma)
        else:
            # predict landmark pos
            lmks_list_dict = predict_RBF_patch_pattern_lmk_pos(lat_pred, np.array(patterns), sigma, lmk_idx)

            # check if we need to add a new pattern
            if not lmks_list_dict[0] or do_force_label:
                print(" - need labeling")
                # label image
                lmk_pos, new_pattern = label_and_construct_patterns(img, lat_pred,
                                                                    im_ratio=im_ratio,
                                                                    k_size=k_size,
                                                                    fig_title=fig_title)

                if lmk_pos is not None:
                    # save labelled image idx
                    patterns.append(new_pattern)
                    label_img_idx.append(i)
                    print("len(patterns)", len(patterns))
                    print("label_img_idx", label_img_idx)

                    # optimize sigma
                    if use_only_last:
                        new_sigma = optimize_sigma(images, np.array(patterns), sigma, [label_img_idx[-1]], init_sigma, lr_rate)
                    else:
                        new_sigma = optimize_sigma(images, np.array(patterns), sigma, label_img_idx, init_sigma, lr_rate)

                    if new_sigma < sigma:
                        sigma = new_sigma
                    print("new sigma", sigma)
            elif len(lmks_list_dict[0]) > 1:
                print("- {} landmarks found!".format(len(lmks_list_dict[0])))
                new_sigma = decrease_sigma(img, np.array(patterns), sigma, lr_rate, lmk_type, config)
                print("old sigma: {}, new sigma: {}".format(sigma, new_sigma))
                sigma = new_sigma
            else:
                print(" - OK")

        print()

    return patterns, sigma
