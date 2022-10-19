import numpy as np
import matplotlib.pyplot as plt
from utils.LMK.InteractiveImage import InteractiveImage


def get_lmk_on_image(image, im_factor=1, pre_processing=None, fig_title=''):
    if pre_processing == 'VGG19':
        # (un-)process image from VGG19 pre-processing
        image = np.array(image + 128) / 256
        image = image[..., ::-1]  # rgb
        image[image > 1] = 1.0

    fig, ax = plt.subplots(figsize=(2.24 * im_factor, 2.24 * im_factor))
    fig.suptitle(fig_title)
    inter_img = InteractiveImage(ax, image)
    inter_img.connect()
    plt.show()

    lmk_pos = None
    if not inter_img.is_occluded:
        lmk_pos = inter_img.press

    return lmk_pos