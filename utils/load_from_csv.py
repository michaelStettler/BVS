import cv2
import os
import numpy as np
from tqdm import tqdm


def load_from_csv(df, config, x_col="image_path"):
    """
    helper function to load data and put them into an numpy array

    Note: consider using a generator
    :return:
    """
    num_data = len(df.index)
    input_shape = config['input_shape']
    im_size = tuple(input_shape)
    target_size = tuple(input_shape[:-1])

    #  declare x and y
    x = np.zeros((num_data, im_size[0], im_size[1], im_size[2]), dtype=np.float16)
    y = np.zeros(num_data, dtype=np.float16)

    # load images from dataframe
    directory = config['directory']
    index = 0
    for _, row in tqdm(df.iterrows(), total=num_data):
        # load img
        im_path = row[x_col]
        if directory is not None:
            im_path = os.path.join(directory, row[x_col])
        im = cv2.imread(im_path)

        if im is None:
            raise ValueError("Image is None, control given path: {}".format(im_path))

        if config.get("crop") is not None:
            crop_idx = config["crop"]
            im = im[crop_idx[0]:crop_idx[1], crop_idx[2]:crop_idx[3]]

        im = cv2.resize(im, target_size)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # transform image into black and white but keep the 3 rgb channels
        if config.get('use_black_n_white') is not None:
            if config['use_black_n_white']:
                im_rgb = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                im_rgb = np.expand_dims(im_rgb, axis=2)
                im_rgb = np.repeat(im_rgb, 3, axis=2)

        x[index, :, :, :] = im_rgb
        y[index] = row['category']
        index += 1

    return [x, y]
