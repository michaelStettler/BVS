import numpy as np
import os

from multiprocessing import Pool

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


def label_and_construct_patterns(img, pred, im_factor=1, k_size=(7, 7), pre_processing='VGG19', fig_title=''):
    # label image
    lmk_pos = get_lmk_on_image(img, im_factor=im_factor, pre_processing=pre_processing, fig_title=fig_title)

    if lmk_pos is not None:
        # construct patterns
        pattern = construct_pattern(pred, lmk_pos, k_size, ratio=224 / 56)
        pattern = np.expand_dims(pattern, axis=0)  # add a dimension to mimic 1 landmark dimension
    else:
        print("no landmark positions!")
        pattern = None

    return lmk_pos, pattern


def construct_RBF_patterns(images, v4_model, lmk_type, config, lr_rate=100, init_sigma=100, im_factor=1, k_size=(7, 7),
                           use_only_last=False, loaded_patterns=None, loaded_sigma=None, train_idx=None, lmk_name='',
                           max_sigma=None):
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
                                                                im_factor=im_factor,
                                                                k_size=k_size,
                                                                fig_title=fig_title)

            if lmk_pos is not None:
                # save labelled image idx
                patterns.append(new_pattern)
                label_img_idx.append(i)

                # optimize sigma
                sigma = optimize_sigma(images, v4_model, lmk_type, config, np.array(patterns),
                                       sigma, label_img_idx, init_sigma, lr_rate,
                                       is_first=True,
                                       max_sigma=max_sigma)
                print("new sigma", sigma)
        else:
            # predict landmark pos
            lmks_list_dict = predict_RBF_patch_pattern_lmk_pos(lat_pred, np.array(patterns), sigma, lmk_idx)

            # check if we need to add a new pattern
            if not lmks_list_dict[0] or do_force_label:
                print(" - need labeling")
                # label image
                lmk_pos, new_pattern = label_and_construct_patterns(img, lat_pred,
                                                                    im_factor=im_factor,
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
                        new_sigma = optimize_sigma(images, v4_model, lmk_type, config, np.array(patterns),
                                                   sigma, [label_img_idx[-1]], init_sigma, lr_rate,
                                                   max_sigma=max_sigma)
                    else:
                        new_sigma = optimize_sigma(images, v4_model, lmk_type, config, np.array(patterns),
                                                   sigma, label_img_idx, init_sigma, lr_rate,
                                                   max_sigma=max_sigma)

                    if new_sigma < sigma:
                        sigma = new_sigma
                    print("new sigma", sigma)
            elif len(lmks_list_dict[0]) > 1:
                print("- {} landmarks found!".format(len(lmks_list_dict[0])))
                new_sigma = decrease_sigma(img, v4_model, np.array(patterns), sigma, lr_rate, lmk_type, config)
                print("old sigma: {}, new sigma: {}".format(sigma, new_sigma))
                sigma = new_sigma
            else:
                print(" - OK")

        print()

    return patterns, sigma


def create_RBF_LMK(config, data, v4_model, n_iter=2, max_sigma=None,
                   FR_patterns=None, FR_sigma=None, FER_patterns=None, FER_sigma=None, save=True):
    FR_patterns_list = []
    FR_sigma_list = []
    FER_patterns_list = []
    FER_sigma_list = []

    patterns = None
    sigma = None

    num_img_per_avatar = int(len(data[0]) / len(config["avatar_types"]))
    avatar_idx = [i * num_img_per_avatar for i in range(len(config["avatar_types"]))]
    print("num_img_per_avatar", num_img_per_avatar)
    print("avatar_idx", avatar_idx)
    # FR lmk pattern
    for a, (avatar, idx) in enumerate(zip(config["avatar_types"], avatar_idx)):
        FR_pat_list = []
        FR_sig_list = []
        for l, lmk_name in enumerate(config["FR_lmk_name"]):
            if FR_patterns is not None:
                patterns = FR_patterns[a][l]
                sigma = FR_sigma[a][l]

            print("avatar: {}, lmk_name: {}, sigma: {}".format(avatar, lmk_name, sigma))
            for i in range(n_iter):
                patterns, sigma = construct_RBF_patterns(data[0][idx:(idx+num_img_per_avatar)], v4_model, "FR", config,
                                                         init_sigma=config["init_sigma"],
                                                         im_factor=3,  # size of image
                                                         k_size=config["k_size"],
                                                         use_only_last=config["use_only_last"],
                                                         loaded_patterns=patterns,
                                                         loaded_sigma=sigma,
                                                         lmk_name=lmk_name,
                                                         max_sigma=max_sigma)

            print("-- Labeling and optimization finished --")
            print("shape patterns", np.shape(patterns))
            print("sigma", sigma)

            FR_pat_list.append(patterns)
            FR_sig_list.append(sigma)

            if save:
                np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FR_{}_patterns_{}".format(avatar, lmk_name)), patterns)
                np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FR_{}_sigma_{}".format(avatar, lmk_name)), sigma)

            patterns = None
            sigma = None

        FR_patterns_list.append(FR_pat_list)
        FR_sigma_list.append(FR_sig_list)

    # FER LMK patterns
    for l, lmk_name in enumerate(config["FER_lmk_name"]):
        if FER_patterns is not None:
            patterns = FER_patterns[l]
            sigma = FER_sigma[l]

        print("lmk_name:", lmk_name)
        for i in range(n_iter):
            patterns, sigma = construct_RBF_patterns(data[0], v4_model, "FER", config,
                                                     init_sigma=config["init_sigma"],
                                                     im_factor=3,  # size of image
                                                     k_size=config["k_size"],
                                                     use_only_last=config["use_only_last"],
                                                     loaded_patterns=patterns,
                                                     loaded_sigma=sigma,
                                                     lmk_name=lmk_name)

        print("-- Labeling and optimization finished --")
        print("shape patterns", np.shape(patterns))
        print("sigma", sigma)

        FER_patterns_list.append(patterns)
        FER_sigma_list.append(sigma)

        if save:
            np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FER_patterns_{}".format(lmk_name)), patterns)
            np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FER_sigma_{}".format(lmk_name)), sigma)

        patterns = None
        sigma = None

    return FR_patterns_list, FR_sigma_list, FER_patterns_list, FER_sigma_list
