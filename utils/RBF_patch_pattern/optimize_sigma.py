import numpy as np
from tqdm import tqdm

from utils.RBF_patch_pattern.lmk_patches import predict_RBF_patch_pattern_lmk_pos


def optimize_sigma_by_landmarks_count(preds, patterns, lmk_idx=None, lr_rate=100, batch_size=16, init_sigmas=None,
                                      act_threshold=0.1, dist_threshold=1.5, patch_size=14, verbose=False,
                                      disable_tqdm=False):

    activities_dict = []
    n_images = np.shape(preds)[0]
    print("shape preds", np.shape(preds))
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
            print("sigma:", sigma, end='\r')
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