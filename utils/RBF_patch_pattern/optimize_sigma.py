import numpy as np
from tqdm import tqdm

from utils.RBF_patch_pattern.lmk_patches import predict_RBF_patch_pattern_lmk_pos
from utils.Semantic.filter_with_semantic_units import get_semantic_pred


def optimize_sigma_by_landmarks_count(preds, patterns, lmk_idx=None, lr_rate=100, batch_size=16, init_sigmas=None,
                                      act_threshold=0.1, dist_threshold=1.5, patch_size=14, verbose=False,
                                      disable_tqdm=False, prev_sigma=None, max_sigma=None):

    activities_dict = []
    n_images = np.shape(preds)[0]
    print("shape patterns", np.shape(patterns))

    if lmk_idx is None:
        lmk_indexes = range(np.shape(patterns)[1])
    else:
        lmk_indexes = lmk_idx

    opt_sigmas = []
    for l in tqdm(lmk_indexes, disable=disable_tqdm):

        do_continue = True

        if init_sigmas is not None:
            sigma = init_sigmas[l]
        else:
            sigma = lr_rate

        memory_max_pool_activity = []  # allows to save the previous step
        memory_sigma = sigma  # allows to save the previous step
        while do_continue:
            print("sigma: {} (max_sigma {})".format(sigma, max_sigma), end='\r')
            # predict lmk positions
            lmks_dict = predict_RBF_patch_pattern_lmk_pos(preds, patterns, sigma, l,
                                                          batch_size=batch_size,
                                                          act_threshold=act_threshold,
                                                          dist_threshold=dist_threshold,
                                                          patch_size=patch_size)

            # compute number of lmk found to continue training
            min_lmk_per_image = 1000
            max_lmk_per_image = 0
            # check if more than one lmk per image
            for i in range(n_images):
                n_lmk = len(lmks_dict[i])

                if n_lmk < min_lmk_per_image:
                    min_lmk_per_image = n_lmk
                if n_lmk > max_lmk_per_image:
                    max_lmk_per_image = n_lmk

            if min_lmk_per_image <= 1 and max_lmk_per_image < 2:
                memory_sigma = sigma
                sigma += lr_rate
                memory_max_pool_activity = lmks_dict

                # stop if sigma gets bigger than a previous one
                if prev_sigma is not None and sigma > prev_sigma:
                    do_continue = False

                # stop if sigma is bigger than a set value
                if max_sigma is not None and sigma > max_sigma:
                    do_continue = False
            else:
                do_continue = False

        if len(memory_max_pool_activity) == 0:
            memory_max_pool_activity = lmks_dict
            print("[WARNING] activities_map_dict might use non-optimized parameters for lmk: {}!".format(l))

        opt_sigmas.append(memory_sigma)
        activities_dict.append(memory_max_pool_activity)

    # transform activites dict to match order of (n_images, n_lmk, n_pos, 2)
    activities_map_dict = []
    for i in range(len(preds)):
        dict = {}
        for l in range(len(activities_dict)):
            if len(activities_dict[l][i]) != 0:
                dict[l] = activities_dict[l][i][0]
        activities_map_dict.append(dict)

    print("sigma:", sigma)

    if verbose:
        print("---------------------------------")
        print("Finish optimizing sigmas")
        print(opt_sigmas)
        print("---------------------------------")
        print("---------------------------------")
        print()

    return activities_map_dict, np.array(opt_sigmas)


def optimize_sigma(images, v4_model, lmk_type, config, patterns, sigma, label_img_idx, init_sigma, lr_rate,
                   is_first=False, max_sigma=None):

    # get all latent image (feature extractions) from the labeled images
    preds = []
    for labeled_idx in label_img_idx:
        im_pred = get_semantic_pred(v4_model, images[labeled_idx], lmk_type, config)
        preds.append(np.squeeze(im_pred))
    preds = np.array(preds)

    if is_first:
        prev_sigma = None
    else:
        prev_sigma = sigma

    # optimize sigma over all labeled images for each pattern
    lmks_dict, opt_sigmas = optimize_sigma_by_landmarks_count(preds, patterns,
                                                              lr_rate=lr_rate,
                                                              init_sigmas=[init_sigma],
                                                              act_threshold=0.1,
                                                              disable_tqdm=True,
                                                              prev_sigma=prev_sigma,
                                                              max_sigma=max_sigma)

    # save sigma
    return opt_sigmas[0]


def get_lmk_distances(lmks_dict):
    # retrieve all positions
    positions = []
    for l in lmks_dict[0]:
        positions.append(lmks_dict[0][l]['pos'])
    positions = np.array(positions)

    # compute distances between all positions as a square matrix
    distances = []
    for p_start in positions:
        for p_end in positions:
            dist = np.linalg.norm(p_start - p_end)
            distances.append(dist)

    return np.array(distances)


def decrease_sigma(image, v4_model, patterns, sigma, lr_rate, lmk_type, config,
                   act_threshold=0.1, dist_threshold=1.5, patch_size=14, max_dist=3):
    im_pred = get_semantic_pred(v4_model, image, lmk_type, config)
    print("shape im_pred", np.shape(im_pred))
    is_too_close = True

    while(is_too_close):
        lmks_dict = predict_RBF_patch_pattern_lmk_pos(im_pred, patterns, sigma, 0,
                                                      act_threshold=act_threshold,
                                                      dist_threshold=dist_threshold,
                                                      patch_size=patch_size)
        # check how many landmarks found
        n_lmk = len(lmks_dict[0])

        # if (still )more than 1, then look if they are too close
        if n_lmk > 1:
            # get distance between all find lmks
            distances = get_lmk_distances(lmks_dict)

            # check if distance is smaller than max distance
            if np.amax(distances) < max_dist:
                is_too_close = False
            else:
                sigma -= lr_rate
            print("sigma: {} (max_dist:{})".format(sigma, np.amax(distances)), end='\r')
        else:
            is_too_close = False
    print("")

    return sigma