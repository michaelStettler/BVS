import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.patches import pred_to_patch
from utils.patches import get_patches_centers
from utils.patches import max_pool_patches_activity
from plots_utils.plot_BVS import display_image
from plots_utils.plot_BVS import display_images
from utils.RBF_pattern.construct_patterns import construct_pattern
from utils.RBF_pattern.construct_patterns import compute_RBF_pattern_activity_maps

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m tests.LMK.t01_optimize_FERG_lmks
"""


def get_latent_image(v4_model, image, lmk_type, config):
    print("get_latent_image, shape image", np.shape(image))
    v4_pred = v4_model.predict(np.expand_dims(image, axis=0), verbose=0)
    print("shape v4_pred", np.shape(v4_pred))

    if lmk_type == 'FR':
        # FR predictions
        print("FR predictions")
        eyes_preds = v4_pred[..., config['best_eyes_IoU_ft']]
        nose_preds = v4_pred[..., config['best_nose_IoU_ft']]
        preds = np.concatenate((eyes_preds, nose_preds), axis=3)
    elif lmk_type == 'FER':
        # FER predictions
        print("FER predictions")
        eyebrow_preds = v4_pred[..., config['best_eyebrow_IoU_ft']]
        lips_preds = v4_pred[..., config['best_lips_IoU_ft']]
        preds = np.concatenate((eyebrow_preds, lips_preds), axis=3)
    else:
        print("lmk_type {} is not implemented".format(lmk_type))

    print("shape v4_pred after filtering", np.shape(preds))

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


def optimize_sigma(images, patterns, threshold):
    print("optimize_delta")

    # compute latent predictions
    preds = []
    for image in images:
        lat_image = np.squeeze(get_latent_image(v4_model, image, lmk_type, config))
        preds.append(lat_image)
    preds = np.array(preds)

    # compute activity maps
    print("shape preds", np.shape(preds))
    print("shape patterns", np.shape(patterns))
    activity_maps = compute_RBF_pattern_activity_maps(preds, patterns)
    print("shape activity_maps", np.shape(activity_maps))

    # threshold activity

    # get center of activity

    # count number of lmk found
    return 0


def optimize(images, v4_model, lmk_type, config, im_ratio=1, filt_size=(3, 3)):
    patterns = []
    label_img_idx = []

    for i, img in enumerate(images):
        print(i, "shape img", np.shape(img))
        # transform image to latent space
        lat_image = get_latent_image(v4_model, img, lmk_type, config)

        # find landmarks
        # if no lmk_pos that means its the first image and so we need a first label
        if len(patterns) == 0:
            # label image
            lmk_pos = get_lmk_on_image(img, im_ratio=im_ratio, pre_processing='VGG19')
            print("lmk_pos", lmk_pos)

            # construct patterns
            if lmk_pos is not None:
                pattern = construct_pattern(lat_image, lmk_pos, (7, 7), ratio=224/56)
                pattern = np.expand_dims(pattern, axis=0)  # add a dimension to mimic 1 landmark dimension
                print("shape pattern", np.shape(pattern))
                patterns.append(pattern)

                # save image idx
                label_img_idx.append(i)

                # optimize sigma over all labeled images for each patterns
                sigma = optimize_sigma(images[label_img_idx], patterns, threshold=0.1)

        else:
            # predict landmark pos
            print("predict lmk pos")
        print()

    return patterns, sigma


if __name__ == '__main__':
    # declare variables
    im_ratio = 3
    filt_size = (3, 3)
    lmk_type = 'FER'

    # define configuration
    config_path = 'LMK_t01_optimize_FERG_lmks_m0001.json'
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # load data
    train_data = load_data(config)
    print("len train_data[0]", len(train_data[0]))
    print("-- Data loaded --")
    print()

    # load feature extraction model
    v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
    v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
    size_ft = tuple(np.shape(v4_model.output)[1:3])
    print("size_ft", size_ft)
    print("-- Extraction Model loaded --")
    print()

    # optimize
    optimize(train_data[0], v4_model, lmk_type, config, im_ratio=im_ratio, filt_size=filt_size)

