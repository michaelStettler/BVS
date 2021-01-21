"""
2021/01/20
This script creates an animation of the cnn response to a moving face.
for 150 frames this takes approximately 10 minutes
"""

import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.plot_cnn_output import plot_cnn_output

# load config
# t0001: human_anger, t0002: human_fear, t0003: monkey_anger, t0004: monkey_fear
config = load_config("norm_base_animate_cnn_response_t0004.json")

# load images
images,_ = load_data(config, train=config["dataset"])

# get cnn response
model = tf.keras.applications.VGG19(include_top=False, weights=config['weights'], input_shape=(224,224,3))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)
response = model.predict(np.array(images))

# make folder
folder = os.path.join("models/saved", config["save_name"])
if not os.path.exists(folder):
    os.mkdir(folder)

# animation
plot_cnn_output(response, folder, config["movie_name"]+".mp4", title=config["plot_title"], image=images, video=True)

