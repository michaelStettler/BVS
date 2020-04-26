import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from bvs.utils.create_examples import *
from bvs.utils.draw_on_grid import draw_on_grid
from bvs.utils.save_debugg_BotUpSaliency_output import save_debugg_BotUp_output
from bvs.layers import GaborFilters
from bvs.layers import BotUpSaliency

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

np.set_printoptions(precision=4, linewidth=150)

print("Visualize saliency")
bypass_gabor_filters = True
debug = False

config = 'configs/config_test2.json'
# image_type = 'half_vert_hori_pattern'
# image_type = 'half_vert_hori_pattern_small'
# image_type = 'fig5.14F'
image_type = 'code_example'
# load data
if image_type == 'monkey':
    img_path = '../data/02_FearGrin_1.0_120fps/02_FearGrin_1.0_120fps.0000.jpeg'  # monkey face
    img = cv2.imread(img_path)
    # crop and resize image
    img = img[:, 280:1000, :]
    img = cv2.resize(img, (256, 256))
elif image_type == 'simple_vertical_bar':
    img = get_simple_vertical_bar()
    config = 'configs/simple_config.json'
elif image_type == 'double_simple_vertical_bar':
    img = get_double_simple_vertical_bar()
    config = 'configs/simple_config.json'
elif image_type == 'half_vert_hori_pattern':
    img = get_half_vert_hori_pattern()
    config = 'configs/simple_config.json'
elif image_type == 'half_vert_hori_pattern_small':
    img = get_half_vert_hori_pattern_small()
    config = 'configs/simple_config.json'
elif image_type == 'fig5.14F':
    img = get_fig_5_14F()
    config = 'configs/simple_config.json'
    bypass_gabor_filters = True
elif image_type == 'code_example':
    img = get_code_example()
    config = 'configs/simple_config.json'
    bypass_gabor_filters = True
elif image_type == 'mnist':
    img = get_mnist()
    config = 'configs/simple_config.json'
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

# build model
input = Input(shape=np.shape(img))

if not bypass_gabor_filters:
    gabor_layer = GaborFilters((15, 15),
                               theta=thetas,
                               lamda=lamdas,
                               gamma=gamma,
                               phi=phi,
                               use_octave=use_octave,
                               octave=octave,
                               per_channel=per_channel,
                               per_color_channel=per_color_channel)
    x = gabor_layer(input)
else:
    x = input

steps = 200
bu_saliency = BotUpSaliency((9, 9),
                            K=n_rot,
                            steps=steps,
                            epsilon=0.1,
                            verbose=2)

x = bu_saliency(x)


model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# predict image
test_img = np.expand_dims(img, axis=0)
pred = model.predict(x=test_img)

# save input
if bypass_gabor_filters:
    input_print = draw_on_grid(img)
else:
    input_print = img.astype(np.uint8)
print("shape input_print", np.shape(input_print))
print("min max input_print", np.min(input_print), np.max(input_print))
cv2.imwrite("bvs/video/input.jpeg", input_print)

# plot saliency
if not debug:
    saliency = pred[0]
    print("shape saliency", np.shape(saliency))
    print("min max saliency", np.min(saliency), np.max(saliency))
    print(saliency)

    # normalize saliency
    saliency = saliency - np.min(saliency)
    saliency = saliency / np.max(saliency)
    saliency_map = np.expand_dims(saliency, axis=2)
    saliency_map = np.array(saliency_map * 255).astype(np.uint8)
    print("shape saliency map", np.shape(saliency_map))
    # saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("bvs/video/V1_saliency_map.jpeg", saliency_map.astype(np.uint8))


else:
    save_debugg_BotUp_output(pred, steps)



