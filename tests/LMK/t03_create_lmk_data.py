import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.get_csv_file_FERG import edit_FERG_csv_file_from_config
from utils.extraction_model import load_extraction_model
from utils.RBF_patch_pattern.lmk_patches import predict_RBF_patch_pattern_lmk_pos

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m tests.LMK.t03_create_lmk_data
"""


def get_latent_pred(v4_model, image, lmk_type, config):
    # et latent prediction
    v4_pred = v4_model.predict(np.expand_dims(image, axis=0), verbose=0)

    # filter feature maps with semantic concepts
    if lmk_type == 'FR':
        # FR predictions
        eyes_preds = v4_pred[..., config['best_eyes_IoU_ft']]
        nose_preds = v4_pred[..., config['best_nose_IoU_ft']]
        preds = np.concatenate((eyes_preds, nose_preds), axis=3)
    elif lmk_type == 'FER':
        # FER predictions
        eyebrow_preds = v4_pred[..., config['best_eyebrow_IoU_ft']]
        lips_preds = v4_pred[..., config['best_lips_IoU_ft']]
        preds = np.concatenate((eyebrow_preds, lips_preds), axis=3)
    else:
        print("lmk_type {} is not implemented".format(lmk_type))

    return preds


def create_lmk_dataset(images, v4_model, lmk_type, config, patterns, sigma):
    lmks_positions = []

    for i, img, in tqdm(enumerate(images), total=len(images)):
        # transform image to latent space
        im_pred = get_latent_pred(v4_model, img, lmk_type, config)

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

            else:
                lmks_pos.append(lmks_dict[0][0]['pos'])

        lmks_positions.append(lmks_pos)

    return np.array(lmks_positions).astype(np.float16)


if __name__ == '__main__':
    # declare variables
    do_load = True
    im_ratio = 3
    k_size = (7, 7)
    lmk_type = 'FER'
    # define avatar
    avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
    avatar_name = avatar_names[2]

    # define configuration
    config_path = 'LMK_t03_create_lmk_data_m0001.json'
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # modify csv according to avatar name
    edit_FERG_csv_file_from_config(config, avatar_name)

    # define loading variables -> add all to config
    lmk_names = ['left_eyebrow_ext', 'left_eyebrow_int', 'right_eyebrow_int', 'right_eyebrow_ext',
                 'left_mouth', 'top_mouth', 'right_mouth', 'down_mouth',
                 'left_eyelid', 'right_eyelid']
    path = config['directory']

    # load data
    train_data = load_data(config)
    test_data = load_data(config, train=False)
    print("-- Data loaded --")
    print("len train_data[0]", len(train_data[0]))
    print("len test_data[0]", len(test_data[0]))
    print()

    # load feature extraction model
    v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
    v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
    size_ft = tuple(np.shape(v4_model.output)[1:3])
    print("-- Extraction Model loaded --")
    print("size_ft", size_ft)
    print()

    # load lmk parameters
    patterns = []
    sigma = []
    for lmk_name in lmk_names:
        patttern = np.load(os.path.join(path, 'saved_patterns', 'patterns_' + avatar_name + '_' + lmk_name + '.npy'))
        patterns.append(patttern)
        sigma.append(
            int(np.load(os.path.join(path, 'saved_patterns', 'sigma_' + avatar_name + '_' + lmk_name + '.npy'))))

    print("-- Loaded optimized patterns finished --")
    print("len patterns", len(patterns))
    print("sigma", sigma)

    # create lmk dataset
    lmk_pos_data = create_lmk_dataset(train_data[0], v4_model, lmk_type, config, patterns, sigma)
    print("shape lmk_pos_data", np.shape(lmk_pos_data))
    np.save(os.path.join(path, 'saved_lmks_pos', avatar_name + "_lmk_pos"), lmk_pos_data)

    # create test lmk dataset
    lmk_pos_data = create_lmk_dataset(test_data[0], v4_model, lmk_type, config, patterns, sigma)
    print("shape test lmk_pos_data", np.shape(lmk_pos_data))
    np.save(os.path.join(path, 'saved_lmks_pos', "test_" + avatar_name + "_lmk_pos"), lmk_pos_data)
