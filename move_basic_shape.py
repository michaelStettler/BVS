"""
2021/01/18
This script plots the network response to basic shapes and the movement of them.
Look under load_data._load_basic_shapes for all the possibilities of basic shapes.
"""

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import os

from utils.plot_cnn_output import plot_cnn_output
from utils.load_config import load_config
from utils.load_data import load_data

# load config
config = load_config("norm_base_basic_shape_t0002.json")

# make folder
folder = os.path.join("models/saved", config["save_name"])
if not os.path.exists(folder):
    os.mkdir(folder)

# cnn
model = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("block3_pool").output)

# calculate and plot response
images = load_data(config, train=config["subset"])
response = model.predict(np.array(images))
for i, image in enumerate(images):
    plot_cnn_output(response[i], os.path.join(folder, config["sub_folder"]), f"plot{i}.png",
                    title="block3_pool response", image=image)