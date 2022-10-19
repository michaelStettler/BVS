import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.get_csv_file_FERG import edit_FERG_csv_file_from_config
from utils.extraction_model import load_extraction_model
from utils.RBF_patch_pattern.construct_patterns import construct_pattern
from utils.RBF_patch_pattern.construct_patterns import construct_RBF_patterns
from utils.RBF_patch_pattern.lmk_patches import predict_RBF_patch_pattern_lmk_pos
from utils.RBF_patch_pattern.optimize_sigma import optimize_sigma_by_landmarks_count
from plots_utils.plot_BVS import display_images

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
        self.is_occluded = False

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
        elif self.press is None and event.key == 'o':
            print("Landmark Occluded")
            self.is_occluded = True
            plt.close()
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


def get_lmk_on_image(image, im_ratio=1, pre_processing=None, fig_title=''):
    if pre_processing == 'VGG19':
        # (un-)process image from VGG19 pre-processing
        image = np.array(image + 128) / 256
        image = image[..., ::-1]  # rgb
        image[image > 1] = 1.0

    fig, ax = plt.subplots(figsize=(2.24 * im_ratio, 2.24 * im_ratio))
    fig.suptitle(fig_title)
    inter_img = InteractiveImag(ax, image)
    inter_img.connect()
    plt.show()

    lmk_pos = None
    if not inter_img.is_occluded:
        lmk_pos = inter_img.press

    return lmk_pos


def label_and_construct_patterns(img, pred, im_ratio=1, k_size=(7, 7), pre_processing='VGG19', fig_title=''):
    # label image
    lmk_pos = get_lmk_on_image(img, im_ratio=im_ratio, pre_processing=pre_processing, fig_title=fig_title)

    if lmk_pos is not None:
        # construct patterns
        pattern = construct_pattern(pred, lmk_pos, k_size, ratio=224 / 56)
        pattern = np.expand_dims(pattern, axis=0)  # add a dimension to mimic 1 landmark dimension
    else:
        print("no landmark positions!")
        pattern = None

    return lmk_pos, pattern


def optimize_sigma(images, patterns, sigma, label_img_idx, init_sigma, lr_rate, is_first=False):
    # get all latent image (feature extractions) from the labeled images
    preds = []
    for labeled_idx in label_img_idx:
        im_pred = get_latent_pred(v4_model, images[labeled_idx], lmk_type, config)
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
                                                              prev_sigma=prev_sigma)

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


def decrease_sigma(image, patterns, sigma, lr_rate, lmk_type, config,
                   act_threshold=0.1, dist_threshold=1.5, patch_size=14, max_dist=3):
    im_pred = get_latent_pred(v4_model, image, lmk_type, config)
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


def count_found_RBF_patterns(images, patterns, sigma, v4_model, lmk_type, config, max_value=3, display_failed_img=False):
    lmk_idx = 0
    n_found = 0
    failed_img_idx = []

    for i, img in tqdm(enumerate(images)):
        # transform image to latent space
        lat_pred = get_latent_pred(v4_model, img, lmk_type, config)

        # predict landmark pos
        lmks_list_dict = predict_RBF_patch_pattern_lmk_pos(lat_pred, patterns, sigma, lmk_idx)

        # count if not empty
        if len(lmks_list_dict[0]) == 0:
            # no lmk found, this is bad
            print("no landmark found on image idx:", i)
            failed_img_idx.append(i)
        elif len(lmks_list_dict[0]) == 1:
            n_found += 1
        else:
            # means we have more than 2
            distances = get_lmk_distances(lmks_list_dict)

            # check if the distance between found lmk are greater than the max_value
            if np.amax(distances) > max_value:
                failed_img_idx.append(i)
                print("distance between landmark found on image idx {} seem too big:".format(i))
            else:
                n_found += 1

    if display_failed_img and len(failed_img_idx) > 0:
        display_images(images[failed_img_idx], pre_processing='VGG19')

    return n_found


if __name__ == '__main__':
    # declare variables
    do_load = True
    do_train = True
    use_only_last = True
    im_ratio = 3
    k_size = (7, 7)
    lmk_type = 'FER'
    init_sigma = 2000
    train_idx = None
    # train_idx = [0]

    # saving variables
    avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
    avatar_name = avatar_names[4]

    lmk_names = ['left_eyebrow_ext', 'left_eyebrow_int', 'right_eyebrow_int', 'right_eyebrow_ext',
                 'left_mouth', 'top_mouth', 'right_mouth', 'down_mouth',
                 'left_eyelid', 'right_eyelid']
    lmk_name = lmk_names[9]
    print("lmk_name:", lmk_name)


    save_path = '/Users/michaelstettler/PycharmProjects/BVS/data/FERG_DB_256/saved_patterns/'
    save_patterns_name = 'patterns_' + avatar_name + '_' + lmk_name
    save_sigma_name = 'sigma_' + avatar_name + '_' + lmk_name

    # define configuration
    config_path = 'LMK_t01_optimize_FERG_lmks_m0001.json'  # mac
    # config_path = 'LMK_t01_optimize_FERG_lmks_w0001.json'  # windows
    # load config
    config = load_config(config_path, path='configs/LMK')
    print("-- Config loaded --")
    print()

    # define path an name
    save_path = config['save_path']
    save_patterns_name = 'patterns_' + avatar_name + '_' + lmk_name
    save_sigma_name = 'sigma_' + avatar_name + '_' + lmk_name

    # modify csv according to avatar name
    edit_FERG_csv_file_from_config(config, avatar_name)

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

    if do_load:
        patterns = np.load(os.path.join(save_path, save_patterns_name + ".npy"))
        sigma = int(np.load(os.path.join(save_path, save_sigma_name + ".npy")))

        print("-- Loaded optimized patterns finished --")
        print("shape patterns", np.shape(patterns))
        print("sigma", sigma)
    else:
        patterns = None
        sigma = None

    if do_train:
        # construct_RBF_patterns
        patterns, sigma = construct_RBF_patterns(train_data[0], v4_model, lmk_type, config,
                                                 init_sigma=init_sigma,
                                                 im_ratio=im_ratio,
                                                 k_size=k_size,
                                                 use_only_last=use_only_last,
                                                 loaded_patterns=patterns,
                                                 loaded_sigma=sigma,
                                                 train_idx=train_idx,
                                                 lmk_name=lmk_name)
        print("-- Labeling and optimization finished --")
        print("shape patterns", np.shape(patterns))
        print("sigma", sigma)

        np.save(os.path.join(save_path, save_patterns_name), patterns)
        np.save(os.path.join(save_path, save_sigma_name), sigma)

        # just to make sure...
        patterns = np.array(patterns)
        sigma = int(sigma)

    # test with test data
    test_data = load_data(config, train=False)
    print("shape test_data[0]", np.shape(test_data[0]))
    n_found = count_found_RBF_patterns(test_data[0], patterns, sigma, v4_model, lmk_type, config,
                                       display_failed_img=True)
    print("n_found: {} (accuracy: {:.2f}%)".format(n_found, (n_found/len(test_data[0]))*100))



