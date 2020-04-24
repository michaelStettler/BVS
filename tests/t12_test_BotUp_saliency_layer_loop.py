import numpy as np
import cv2
import json
from bvs.utils.create_examples import *
from bvs.utils.draw_on_grid import draw_on_grid
from bvs.utils.save_debugg_BotUpSaliency_output import save_debugg_BotUp_output
from bvs.layers import BotUpSaliency

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

print("Visualize saliency")

config = 'configs/config_test2.json'
# image_type = 'half_vert_hori_pattern'
# image_type = 'half_vert_hori_pattern_small'
image_type = 'code_example'
# image_type = 'fig5.14F'
# load data
if image_type == 'code_example':
    img = code_example()
    config = 'configs/simple_config.json'
    custom_img_size = True
if image_type == 'fig5.14F':
    img = get_fig_5_14F()
    config = 'configs/simple_config.json'
    custom_img_size = True
else:
    raise NotImplementedError("Please select a valid image_type to test!")

# load config
with open(config) as f:
  config = json.load(f)

n_rot = config['n_rot']
thetas = np.array(range(n_rot)) * np.pi / n_rot
lamdas = np.array(config['lamdas']) * np.pi
gamma = config['gamma']
phi = np.array(config['phi']) * np.pi
use_octave = config['use_octave']
octave = config['octave']
per_channel = config['per_channel']
per_color_channel = config['per_color_channel']

print("Num orientations {}, lambda {}, gamma {}".format(config['n_rot'], config['lamdas'], config['gamma']))

# save input
input_print = draw_on_grid(img)
cv2.imwrite("bvs/video/input.jpeg", input_print)

for t in range(1):
    print()
    print("-----------------------------------------")
    print("t", t)
    # build model
    inp = Input(shape=np.shape(img))
    bu_saliency = BotUpSaliency((15, 15),
                                K=n_rot,
                                steps=t + 1,
                                epsilon=0.1,
                                verbose=0)

    x = bu_saliency(inp)

    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # predict image
    test_img = np.expand_dims(img, axis=0)
    pred = model.predict(x=test_img)

    save_debugg_BotUp_output(pred, t, custom_img_size)



