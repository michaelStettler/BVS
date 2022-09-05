import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.RBF_patch_pattern.lmk_patches import predict_RBF_patch_pattern_lmk_pos
from plots_utils.plot_BVS import display_image

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m tests.LMK.t02_visualize_RBF_patch_pattern_preds
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


def convert_dict_to_pos(dict, n_lmk=1):
    input = np.zeros((len(dict), n_lmk, 2))
    for f, frame in enumerate(dict):
        for l in range(n_lmk):
            if frame.get(l) is not None:
                input[f, frame[l]['type']] = frame[l]['pos']

    return input


def predict_and_visualize_RBF_patterns(images, patterns_files, sigma_files, v4_model, lmk_type, config, im_ratio=1):
    lmk_idx = 0

    for i, img in enumerate(images):
        print("image:", i)
        # transform image to latent space
        lat_pred = get_latent_pred(v4_model, img, lmk_type, config)

        lmks_pos = []
        for l in range(len(patterns_files)):
            patterns = np.load(os.path.join(config['patterns_path'], patterns_files[l]))
            sigma = int(np.load(os.path.join(config['sigma_path'], sigma_files[l])))

            # predict landmark pos
            lmks_list_dict = predict_RBF_patch_pattern_lmk_pos(lat_pred, patterns, sigma, lmk_idx)
            print("lmks_list_dict lmk:", l, lmks_list_dict)

            # transform dict to pos array
            lmk_pos = convert_dict_to_pos(lmks_list_dict)
            ft2img_ratio = 224/56
            lmk_pos *= ft2img_ratio

            lmks_pos.append(lmk_pos)

        print()
        # display image
        display_image(img, lmks=np.reshape(lmks_pos, (-1, 2)), pre_processing='VGG19', lmk_size=3)
        plt.show()


if __name__ == '__main__':
    # declare variables
    im_ratio = 3
    lmk_type = 'FER'

    patterns_files = ['patterns_jules_left_eyebrow_ext.npy', 'patterns_jules_left_eyebrow_int.npy',
                      'patterns_jules_right_eyebrow_int.npy', 'patterns_jules_right_eyebrow_ext.npy',
                      'patterns_jules_left_mouth.npy', 'patterns_jules_top_mouth.npy',
                      'patterns_jules_right_mouth.npy', 'patterns_jules_down_mouth.npy',
                      'patterns_jules_left_eyelid.npy']
    sigma_files = ['sigma_jules_left_eyebrow_ext.npy', 'sigma_jules_left_eyebrow_int.npy',
                   'sigma_jules_right_eyebrow_int.npy', 'sigma_jules_right_eyebrow_ext.npy',
                   'sigma_jules_left_mouth.npy', 'sigma_jules_top_mouth.npy',
                   'sigma_jules_right_mouth.npy', 'sigma_jules_down_mouth.npy',
                   'sigma_jules_left_eyelid.npy']

    # define configuration
    config_path = 'LMK_t02_visualize_RBF_patch_pattern_preds_m0001.json'
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # load data
    train_data = load_data(config, train=False)
    print("-- Data loaded --")
    print("len train_data[0]", len(train_data[0]))
    print()

    # load feature extraction model
    v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
    v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
    size_ft = tuple(np.shape(v4_model.output)[1:3])
    print("-- Extraction Model loaded --")
    print("size_ft", size_ft)
    print()

    # construct_RBF_patterns
    predict_and_visualize_RBF_patterns(train_data[0], patterns_files, sigma_files, v4_model, lmk_type, config, im_ratio=im_ratio)
    print("-- Predict and Visualize finished --")