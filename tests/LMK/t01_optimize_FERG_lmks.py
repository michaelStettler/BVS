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

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m tests.LMK.t01_optimize_FERG_lmks
"""


class LMKImageOptimizer:

    def __init__(self, ax, image, pre_processing=None, lmk_size=3):
        self.ax = ax
        self.img = image
        self.press = None
        self.ols = int(lmk_size/2)  # offset lmk_size

        if pre_processing == 'VGG19':
            # (un-)process image from VGG19 pre-processing
            self.img = np.array(self.img + 128) / 256
            self.img = self.img[..., ::-1]  # rgb
            self.img[self.img > 1] = 1.0
        print("shape image", np.shape(image))

        self.ax.imshow(self.img)

    def on_click(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        self.press = [np.round(event.xdata).astype(int),
                      np.round(event.ydata).astype(int)]
        print("self.press", self.press)

    def on_release(self, event):
        print("on release")
        if self.press is not None:
            img = np.copy(self.img)

            lmk_x = self.press[1]
            lmk_y = self.press[0]

            img[(lmk_x-self.ols):(lmk_x+self.ols),
                (lmk_y-self.ols):(lmk_y+self.ols)] = [1, 0, 0]

            self.ax.imshow(img)
            plt.draw()

    def on_enter(self, event):
        print("prout", event.key)

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


if __name__ == '__main__':
    # declare variables
    lmk_pos = []
    im_ratio = 3

    # define configuration
    config_path = 'LMK_t01_optimize_FERG_lmks_m0001.json'

    # load config
    config = load_config(config_path, path='configs/LMK')

    # load data
    train_data = load_data(config)
    print("len train_data[0]", len(train_data[0]))

    # optimize
    fig, ax = plt.subplots(figsize=(2.24 * im_ratio, 2.24 * im_ratio))
    if len(lmk_pos) == 0:
        opt = LMKImageOptimizer(ax, train_data[0][0], pre_processing='VGG19')
        opt.connect()

    plt.show()
