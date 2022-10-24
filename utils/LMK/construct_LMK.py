import numpy as np
from tqdm import tqdm

from utils.Semantic.filter_with_semantic_units import get_semantic_pred
from utils.RBF_patch_pattern.lmk_patches import predict_RBF_patch_pattern_lmk_pos


def create_lmk_dataset(images, v4_model, lmk_type, config, patterns, sigma):
    lmks_positions = []

    for i, img, in tqdm(enumerate(images), total=len(images)):
        # transform image to latent space
        im_pred = get_semantic_pred(v4_model, img, lmk_type, config)

        lmks_pos = []
        for l in range(len(patterns)):
            lmks_dict = predict_RBF_patch_pattern_lmk_pos(im_pred, patterns[l], sigma[l], 0,
                                                          act_threshold=config['activity_threshold'],
                                                          dist_threshold=config['distance_threshold'],
                                                          patch_size=config['patch_size'])

            # get highest lmk val if more than one (max_pooling)
            if len(lmks_dict[0]) > 1:
                # print("more than one landmark found in image {} for landmark {}:".format(i, l, lmks_dict[0]))

                max_val_lmk = 0
                max_val_idx = None
                for l_idx in lmks_dict[0]:
                    if lmks_dict[0][l_idx]['max'] > max_val_lmk:
                        max_val_lmk = lmks_dict[0][l_idx]['max']
                        max_val_idx = l_idx
                lmks_pos.append(lmks_dict[0][max_val_idx]['pos'])

            elif len(lmks_dict[0]) == 1:
                lmks_pos.append(lmks_dict[0][0]['pos'])
            else:
                lmks_pos.append([-1, -1])

        lmks_positions.append(lmks_pos)

    return np.array(lmks_positions).astype(np.float16)


def get_identity_and_pos(data, v4_model, config, patterns, sigma):
    FR_pos = []
    identities = []
    positions = []

    n_identitiy = len(patterns)
    n_lmk = len(patterns[0])
    # for i, img, in enumerate(train_data[0][3750:3760]):
    for i, img, in enumerate(data):
        print("image: ", i, end="\r")
        preds = get_semantic_pred(v4_model, img, "FR", config)

        # get all lmk for the image
        lmk_dict_list = []
        n_lmk_type = 0
        for k in range(n_identitiy):
            for l in range(n_lmk):
                lmk_idx = k * n_lmk + l
                FR_dict = predict_RBF_patch_pattern_lmk_pos(preds, patterns[k][l], sigma[k][l], 0)

                # add lmk type (for some reason the lmk_type of predict RBF doesn't work)
                for m in range(len(FR_dict[0])):
                    # change type
                    FR_dict[0][m]['type'] = lmk_idx
                    # happen to dictionary
                    lmk_dict_list.append(FR_dict[0][m])

        if len(lmk_dict_list) < 3:
            print("PROBLEM!!!!! image:", i)

        # max pool the three best activity (for each type = assume one face)
        lmk_type_counter = np.zeros(n_lmk)
        max_pooled_lmk_dict_list = []
        for _ in range(3):
            max_val = 0
            max_lmk = None
            max_idx = None
            max_l_type = None

            # max pooling over type and value
            for l, lmk in enumerate(lmk_dict_list):
                l_type = lmk['type'] % n_lmk
                # keep only if the max value is higher and if the type has not been used yet
                if lmk['max'] > max_val and lmk_type_counter[l_type] == 0:
                    max_val = lmk['max']
                    max_l_type = l_type
                    max_lmk = lmk
                    max_idx = l

            max_pooled_lmk_dict_list.append(max_lmk)
            lmk_type_counter[max_l_type] = 1
            del lmk_dict_list[max_idx]

        if len(max_pooled_lmk_dict_list) < 3:
            print("PROBLEM after pooling!!!!! image:", i)

        # count types of lmk
        id_scores = []
        pos = []
        for k in range(n_identitiy):
            score = 0
            for lmk in max_pooled_lmk_dict_list:
                # store by id type [0 1 2] - [3 4 5]
                id_type_array = [idx for idx in range(k * n_lmk, k * n_lmk + n_lmk)]
                if lmk['type'] in id_type_array:
                    score += 1

                # store positions
                if k == 0:
                    pos.append(lmk['pos'])

            id_scores.append(score)

        identities.append(np.argmax(id_scores))

        # get positions
        pos = np.array(pos)
        mean_x = np.mean([np.amin(pos[:, 0]), np.amax(pos[:, 0])])
        mean_y = np.mean([np.amin(pos[:, 1]), np.amax(pos[:, 1])])

        FR_pos.append(pos)
        positions.append([mean_x, mean_y])

    positions = np.array(positions)

    return np.array(FR_pos), identities, positions
