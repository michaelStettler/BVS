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
