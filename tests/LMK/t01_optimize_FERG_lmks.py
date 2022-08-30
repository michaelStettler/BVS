import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.RBF_patch_pattern.construct_patterns import construct_pattern
from utils.RBF_patch_pattern.lmk_patches import predict_RBF_patch_pattern_lmk_pos
from utils.RBF_patch_pattern.optimize_sigma import optimize_sigma_by_landmarks_count

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m tests.LMK.t01_optimize_FERG_lmks
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


class InteractiveImag:

    def __init__(self, ax, image, lmk_size=3):
        self.ax = ax
        self.img = image
        self.press = None
        self.ols = int(lmk_size/2)  # offset lmk_size

        self.ax.imshow(self.img)

    def on_click(self, event):
        self.press = [np.round(event.xdata).astype(int),
                      np.round(event.ydata).astype(int)]

    def on_release(self, event):
        if self.press is not None:
            img = np.copy(self.img)

            lmk_x = self.press[1]
            lmk_y = self.press[0]

            img[(lmk_x-self.ols):(lmk_x+self.ols),
                (lmk_y-self.ols):(lmk_y+self.ols)] = [1, 0, 0]

            self.ax.imshow(img)
            plt.draw()

    def on_enter(self, event):
        if self.press is not None and event.key == 'enter':
            plt.close()
        elif self.press is None and event.key == 'enter':
            print("No Landmark selected!")
        elif event.key == 'escape':
            plt.close()

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.ax.figure.canvas.mpl_connect(
            'button_press_event', self.on_click)
        self.cidrelease = self.ax.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidenter = self.ax.figure.canvas.mpl_connect(
            'key_press_event', self.on_enter)

    def disconnect(self):
        """Disconnect all callbacks."""
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidenter)


def get_lmk_on_image(image, im_ratio=1, pre_processing=None):
    if pre_processing == 'VGG19':
        # (un-)process image from VGG19 pre-processing
        image = np.array(image + 128) / 256
        image = image[..., ::-1]  # rgb
        image[image > 1] = 1.0

    fig, ax = plt.subplots(figsize=(2.24 * im_ratio, 2.24 * im_ratio))
    inter_img = InteractiveImag(ax, image)
    inter_img.connect()
    plt.show()

    return inter_img.press


def label_and_construct_patterns(img, pred, im_ratio=1, k_size=(7, 7), pre_processing='VGG19'):
    # label image
    lmk_pos = get_lmk_on_image(img, im_ratio=im_ratio, pre_processing=pre_processing)

    if lmk_pos is not None:
        # construct patterns
        pattern = construct_pattern(pred, lmk_pos, k_size, ratio=224 / 56)
        pattern = np.expand_dims(pattern, axis=0)  # add a dimension to mimic 1 landmark dimension
    else:
        raise ValueError("no landmark position!")

    return lmk_pos, pattern


def optimize_sigma(images, patterns, label_img_idx, init_sigma):
    # get all latent image (feature extractions) from the labeled images
    preds = []
    for labeled_idx in label_img_idx:
        im_pred = get_latent_pred(v4_model, images[labeled_idx], lmk_type, config)
        preds.append(np.squeeze(im_pred))
    preds = np.array(preds)

    # optimize sigma over all labeled images for each pattern
    lmks_dict, opt_sigmas = optimize_sigma_by_landmarks_count(preds, patterns,
                                                              init_sigmas=[init_sigma],
                                                              act_threshold=0.1)

    # save sigma
    return opt_sigmas[0]


def construct_RBF_patterns(images, v4_model, lmk_type, config, init_sigma=100, im_ratio=1, k_size=(7, 7)):
    patterns = []
    sigma = init_sigma
    label_img_idx = []
    lmk_idx = 0

    for i, img in enumerate(images):
        print("image: ", i, end='')
        # transform image to latent space
        lat_pred = get_latent_pred(v4_model, img, lmk_type, config)

        # label first image if no patterns
        if len(patterns) == 0:
            print(" - need labeling")
            # label image
            lmk_pos, new_pattern = label_and_construct_patterns(img, lat_pred, im_ratio=im_ratio, k_size=k_size)

            # save labelled image idx
            patterns.append(new_pattern)
            label_img_idx.append(i)

            # optimize sigma
            sigma = optimize_sigma(images, np.array(patterns), label_img_idx, init_sigma)
            print("new sigma", sigma)
        else:
            # predict landmark pos
            lmks_list_dict = predict_RBF_patch_pattern_lmk_pos(lat_pred, np.array(patterns), sigma, lmk_idx)

            # check if we need to add a new pattern
            if not lmks_list_dict[0]:
                print(" - need labeling")
                # label image
                lmk_pos, new_pattern = label_and_construct_patterns(img, lat_pred, im_ratio=im_ratio, k_size=k_size)

                # save labelled image idx
                patterns.append(new_pattern)
                label_img_idx.append(i)
                print("len(patterns)", len(patterns))
                print("label_img_idx", label_img_idx)

                # optimize sigma
                sigma = optimize_sigma(images, np.array(patterns), label_img_idx, init_sigma)
                print("new sigma", sigma)
            else:
                print(" - OK")

        print()

    return patterns, sigma


if __name__ == '__main__':
    # declare variables
    im_ratio = 3
    k_size = (7, 7)
    lmk_type = 'FER'
    avatar_name = 'jules'
    lmk_name = 'left_eyebrow_ext'

    # define configuration
    config_path = 'LMK_t01_optimize_FERG_lmks_m0001.json'
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # load data
    train_data = load_data(config)
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
    patterns, sigma = construct_RBF_patterns(train_data[0], v4_model, lmk_type, config, init_sigma=100, im_ratio=im_ratio, k_size=k_size)
    print("-- Labeling and optimization finished --")
    print("shape patterns", np.shape(patterns))
    print("sigma", sigma)

    save_path = '/Users/michaelstettler/PycharmProjects/BVS/data/FERG_DB_256/saved_patterns/'
    save_patterns_name = 'patterns_' + avatar_name + '_' + lmk_name
    save_sigma_name = 'sigma_' + avatar_name + '_' + lmk_name
    np.save(os.path.join(save_path, save_patterns_name), patterns)
    np.save(os.path.join(save_path, save_sigma_name), sigma)


