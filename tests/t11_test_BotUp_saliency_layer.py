import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from bvs.utils.create_examples import *
from bvs.utils.draw_on_grid import draw_on_grid
from bvs.layers import GaborFilters
from bvs.layers import BotUpSaliency

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

print("Visualize saliency")
bypass_gabor_filters = False

config = 'configs/config_test2.json'
# image_type = 'half_vert_hori_pattern'
# image_type = 'half_vert_hori_pattern_small'
image_type = 'fig5.14F'
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
elif image_type == 'mnist':
    img = get_mnist()
    config = 'configs/simple_config.json'
elif image_type == 'fig5.14F':
    img = get_fig_5_14F()
    config = 'configs/simple_config.json'
    bypass_gabor_filters = True
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
    x = gabor_layer(x)
else:
    x = input

bu_saliency = BotUpSaliency((15, 15),
                            K=n_rot,
                            steps=32,
                            epsilon=0.1,
                            verbose=0)

x = bu_saliency(x)


model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

test_img = np.expand_dims(img, axis=0)
pred = model.predict(x=test_img)
print("shape pred", np.shape(pred))

# save input
if bypass_gabor_filters:
    input_print = draw_on_grid(img)
else:
    input_print = img.astype(np.uint8)
cv2.imwrite("bvs/video/input.jpeg", input_print)

# plot saliency
saliency = pred[0]
print("shape saliency", np.shape(saliency))
print("min max saliency", np.min(saliency), np.max(saliency))
saliency = saliency - np.min(saliency)
saliency = saliency / np.max(saliency)
saliency_map = np.expand_dims(saliency, axis=2)
saliency_map = np.array(saliency_map * 255).astype(np.uint8)
print("shape saliency map", np.shape(saliency_map))
# saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/V1_saliency_map.jpeg", saliency_map.astype(np.uint8))

# # control layer -> uncomment the return line of the BotUpSaliency call method
# # ----------------------------------------------------------------------------------------------------------------------
# I_i_theta = pred[0][0]
# x = pred[1][0]
# gx = pred[2][0]
# i_norm = pred[3][0]
# inhibs = pred[4][0]
# excits = pred[5][0]
# y = pred[6][0]
# gy = pred[7][0]
# inhibs_psi = pred[8][0]
# x_inhib = pred[9][0]
# x_excit = pred[10][0]
# print("shape I_i_theta", np.shape(I_i_theta))
# print("min max I_i_theta", np.min(I_i_theta), np.max(I_i_theta))
# print("shape x", np.shape(x))
# print("min max x", np.min(x), np.max(x))
# print("shape gx", np.shape(gx))
# print("min max gx", np.min(gx), np.max(gx))
# print("shape i_norm", np.shape(i_norm))
# print("shape inhibs", np.shape(inhibs))
# print("shape excits", np.shape(excits))
# print("shape y", np.shape(y))
# print("min max y", np.min(y), np.max(y))
# print("shape gy", np.shape(gy))
# print("min max gy", np.min(gy), np.max(gy))
# print("shape inhibs_psi", np.shape(inhibs_psi))
# print("min max inhibs_psi", np.min(inhibs_psi), np.max(inhibs_psi))
#
# max_column = 6
#
# num_filters = np.shape(I_i_theta)[-1]
# num_column = min(num_filters, max_column)
# num_row = math.ceil(num_filters / num_column)
# x_print = np.expand_dims(I_i_theta, axis=2)
# multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a0_00_I_i_theta.jpeg", heatmap.astype(np.uint8))
#
# # save x activations
# num_filters = np.shape(x)[-1]
# num_column = min(num_filters, max_column)
# num_row = math.ceil(num_filters / num_column)
# x_print = np.expand_dims(x, axis=2)
# multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a01_x.jpeg", heatmap.astype(np.uint8))
#
# # save gx activations
# gx_print = np.expand_dims(gx, axis=2)
# multi_frame = create_multi_frame(gx_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a02_gx(x)_response.jpeg", heatmap.astype(np.uint8))
#
# # save i_norm
# i_norm_print = np.expand_dims(i_norm, axis=2)
# i_norm_print = np.array(i_norm_print).astype(np.uint8)
# multi_frame = create_multi_frame(i_norm_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a03_i_norm.jpeg", heatmap.astype(np.uint8))
#
# # save inhibition
# inhibs_print = np.expand_dims(inhibs, axis=2)
# inhibs_print = np.array(inhibs_print).astype(np.uint8)
# num_filters = np.shape(inhibs)[-1]
# num_column = min(num_filters, max_column)
# num_row = math.ceil(num_filters / num_column)
# multi_frame = create_multi_frame(inhibs_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a04_inibition_response.jpeg", heatmap.astype(np.uint8))
#
# # save excitation
# excits_print = np.expand_dims(excits, axis=2)
# excits_print = np.array(excits_print).astype(np.uint8)
# multi_frame = create_multi_frame(excits_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a05_exitatory_response.jpeg", heatmap.astype(np.uint8))
#
# # print y neuronal response (inhibitory)
# y_print = np.expand_dims(y, axis=2)
# multi_frame = create_multi_frame(y_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a06_y_responses.jpeg", heatmap.astype(np.uint8))
#
# # print gy(y)
# gy_print = np.expand_dims(gy, axis=2)
# multi_frame = create_multi_frame(gy_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a07_gy(y)_response.jpeg", heatmap.astype(np.uint8))
#
# # save inhib psi
# inhibs_psi_print = np.expand_dims(inhibs_psi, axis=2)
# multi_frame = create_multi_frame(inhibs_psi_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a08_inibition_psi.jpeg", heatmap.astype(np.uint8))
#
# # plot inhib response
# x_inhib_print = np.expand_dims(x_inhib, axis=2)
# multi_frame = create_multi_frame(x_inhib_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a09_x_inhib_response.jpeg", heatmap.astype(np.uint8))
#
# # plot excit response
# x_excit_print = np.expand_dims(x_excit, axis=2)
# multi_frame = create_multi_frame(x_excit_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/a10_x_excit_response.jpeg", heatmap.astype(np.uint8))



