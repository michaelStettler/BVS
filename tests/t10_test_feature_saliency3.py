import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from bvs.layers import GaborFilters
from bvs.layers import BotUpSaliency
from bvs.utils.create_preds_seq import create_multi_frame
from bvs.utils.create_preds_seq import create_multi_frame_heatmap
from bvs.utils.create_preds_seq import create_multi_frame_from_multi_channel

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

print("Visualize saliency")

config = 'configs/config_test2.json'
image_type = 'simple_vertical_bar'
# load data
if image_type == 'monkey':
    img_path = '../data/02_FearGrin_1.0_120fps/02_FearGrin_1.0_120fps.0000.jpeg'  # monkey face
    img = cv2.imread(img_path)
    # crop and resize image
    img = img[:, 280:1000, :]
    img = cv2.resize(img, (256, 256))
elif image_type == 'simple_vertical_bar':
    img = np.zeros((50, 50, 1))
    img[10:40, 25, :] = 255
    config = 'configs/simple_config.json'
else:
    raise NotImplementedError("Please select a valid image_type to test!")

# load config
# config = 'config.json'
with open(config) as f:
  config = json.load(f)

# # test case img
# img = np.zeros((256, 256, 3))
# img[:128, :128, 0] = 255
# img[128:, 128:, 1] = 255
# plt.imshow(img)

print("Num orientations {}, lambda {}, gamma {}".format(config['n_rot'], config['lamdas'], config['gamma']))

# build model
input = Input(shape=np.shape(img))

n_rot = config['n_rot']
thetas = np.array(range(n_rot)) * np.pi / n_rot
lamdas = np.array(config['lamdas']) * np.pi
gamma = config['gamma']
phi = np.array(config['phi']) * np.pi
use_octave = config['use_octave']
octave = config['octave']
per_channel = config['per_channel']
per_color_channel = config['per_color_channel']
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

bu_saliency = BotUpSaliency((15, 15),
                            K=n_rot,
                            verbose=4)
# x = bu_saliency(x)

kernels = gabor_layer.kernel
num_kernels = np.shape(kernels)[-1]
print("shape kernels", np.shape(kernels))
max_column = 6
num_column = min(num_kernels, max_column)
num_row = math.ceil(num_kernels / num_column)
print("num column", num_column, "num_row", num_row)
multi_frame = create_multi_frame(kernels, num_row, num_column, (256, 256))
cv2.imwrite("bvs/video/gabor_filters.jpeg", multi_frame.astype(np.uint8))
# gb0 = kernels[:, :, :, 0]
# gb0 = np.expand_dims(gb0, axis=3)
# print("shape gb0", np.shape(gb0))
# gb0 = create_multi_frame(gb0, 1, 1, (256, 256))
# cv2.imwrite("bvs/video/gabor_filter_hori.jpeg", gb0.astype(np.uint8))


model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

test_img = np.expand_dims(img, axis=0)
pred = model.predict(x=test_img)
print("shape pred", np.shape(pred))

activations = pred[0]
print("shape activations", np.shape(activations))

num_activations = np.shape(activations)[-1]
num_column = min(num_activations, max_column)
num_row = math.ceil(num_activations / num_column)
print("num column", num_column, "num_row", num_row)
multi_frame = create_multi_frame_heatmap(img, activations, num_row, num_column, (np.shape(img)[1], np.shape(img)[0]))

# plt.figure()
# plt.imshow(multi_frame)
# plt.show()
cv2.imwrite("bvs/video/heatmap_gabor_filters.jpeg", multi_frame.astype(np.uint8))

# print("shape activations", np.shape(activations))
# saliency_type = "max"
# if saliency_type == 'max':
#     saliency = activations.max(axis=2)
# elif saliency_type == 'sum':
#     saliency = np.sum(activations, axis=2)
# elif saliency_type == "ReLu_sum":
#     activations[activations < 0] = 0
#     saliency = np.sum(activations, axis=2)
# else:
#     raise NotImplementedError("please chose a valid saliency type!")
#
# # normalize saliency
# saliency = saliency - np.min(saliency)
# saliency = saliency / np.max(saliency)
# saliency *= 255
# print("shape saliency", np.shape(saliency))
# cv2.imwrite("bvs/video/saliency.jpeg", saliency.astype(np.uint8))
#
# # --------------------- Zhaoping li's V1 Saliency Model --------------------- #
# # start from activation of each filter
# # normalize outputs
# activations = activations - np.min(activations)
# activations = activations / np.max(activations)
# activations = np.expand_dims(activations, axis=0)
#
# Ic_control = 0
# Ic = 1 + Ic_control
# alphaX = 1
# alphaY = 1
# J0 = 0.8
# mirror_idx = 2
##
# # np.set_printoptions(precision=3, linewidth=200)
# # for dp in range(n_rot):
# #     print()
# #     print(J[:, :, 0, dp])
# #     print("theta prime:", dp * np.pi / n_rot / np.pi * 180)
#
# # # save W inhibition filters
# # idx = 11
# # W_print = W[:, :, idx, :]
# # W_print = np.expand_dims(W_print, axis=2)
# # num_filters = np.shape(W)[-1]
# # num_column = min(num_filters, max_column)
# # num_row = math.ceil(num_filters / num_column)
# # print("shape W_print", np.shape(W_print))
# # multi_frame = create_multi_frame(W_print, num_row, num_column, (256, 256))
# # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# # cv2.imwrite("bvs/video/W"+str(idx)+"_inibition_filter.jpeg", heatmap.astype(np.uint8))
# #
# # # save J exitatory filters
# # J_print = J[:, :, idx, :]
# # J_print = np.expand_dims(J_print, axis=2)
# # multi_frame = create_multi_frame(J_print, num_row, num_column, (256, 256))
# # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# # cv2.imwrite("bvs/video/J"+str(idx)+"_exitatory_filter.jpeg", heatmap.astype(np.uint8))
#
# # save all W inhibition filters
# num_input_channel = np.shape(W)[-2]
# num_filters = num_input_channel * np.shape(W)[-1]
# num_column = min(num_filters, max_column)
# num_row = math.ceil(num_filters / num_column)
# multi_frame = create_multi_frame_from_multi_channel(W, num_row, num_column, (256, 256), num_input_channel)
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/W_inibition_filter.jpeg", heatmap.astype(np.uint8))
#
# # save J exitatory filters
# multi_frame = create_multi_frame_from_multi_channel(J, num_row, num_column, (256, 256), num_input_channel)
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/J_exitatory_filter.jpeg", heatmap.astype(np.uint8))
#
# # start dynamic
# # ----------------------------------------------------------------------------------------------------------------------
# # save activations
# save_intermediate_img = False
# I_i_theta = activations
# print("[declaration] shape I_i_theta", np.shape(I_i_theta))
# # x = activations.copy() + 0.5  # todo x0 as activations, zeros or ones ?
# x = np.zeros(np.shape(activations))
# y = np.zeros(np.shape(activations))
# i_norm_k = np.ones((5, 5, n_rot, n_rot))
# print("[declaration] shape x", np.shape(x), "min max x", np.min(x), np.max(x))
# print("[declaration] shape y", np.shape(y), "min max y", np.min(y), np.max(y))
# print()
#
# num_filters = np.shape(I_i_theta)[-1]
# num_column = min(num_filters, max_column)
# num_row = math.ceil(num_filters / num_column)
# x_print = np.expand_dims(I_i_theta[0], axis=2)
# multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/0_00_I_i_theta.jpeg", heatmap.astype(np.uint8))
#
# # reshape W and J
# print("shape W", np.shape(W))
# W = np.expand_dims(np.moveaxis(W, 2, 0), axis=4)
# J = np.expand_dims(np.moveaxis(J, 2, 0), axis=4)
# print("shape W", np.shape(W))
#
# # epsilon = 0.01
# epsilon = 0.1
# for t in range(15):
#     print()
#     print("----------------------------------------------------------")
#     print("t", t)
#     # i_noise_y = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) / 10 + 0.1
#     # i_noise_x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) / 10 + 0.1
#     i_noise_y = 0
#     i_noise_x = 0
#
#     if save_intermediate_img:
#         num_filters = np.shape(x)[-1]
#         num_column = min(num_filters, max_column)
#         num_row = math.ceil(num_filters / num_column)
#         x_print = np.expand_dims(x[0], axis=2)
#         multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_01_x.jpeg", heatmap.astype(np.uint8))
#         # save gx activations
#         gx_print = np.expand_dims(gx(x)[0], axis=2)
#         multi_frame = create_multi_frame(gx_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_02_gx(x)_response.jpeg", heatmap.astype(np.uint8))
#
#     # convolutions
#     print("[convolution] shape W, np.shape(W)", np.shape(W))
#     print("[convolution] shape gx(activations)", np.shape(gx(x)))
#     print("[convolution] min max gx(activations)", np.min(gx(x)), np.max(gx(x)))
#     i_norm = 0.85 - 2 * np.power(tf.nn.conv2d(gx(x), i_norm_k, strides=1, padding='SAME') / (np.shape(i_norm_k)[0]**2), 2)
#     print("[convolution] 03")
#     print("[convolution] shape i_norm", np.shape(i_norm))
#     print("[convolution] min max i_norm", np.min(i_norm), np.max(i_norm))
#     inhibs = []
#     excits = []
#     for i in range(n_rot):
#         # W_print = W[i]
#         # J_print = J[i]
#         # print("shape W print", np.shape(W_print))
#
#         inhib = tf.nn.conv2d(gx(x), W[i], strides=1, padding='SAME')
#         excit = tf.nn.conv2d(gx(x), J[i], strides=1, padding='SAME')
#         inhibs.append(inhib)
#         excits.append(excit)
#
#         # # print filter
#         # W_print = np.swapaxes(W_print, 2, 3)
#         # print("shape W print", np.shape(W_print))
#         # num_filters = np.shape(W_print)[-1]
#         # num_column = min(num_filters, max_column)
#         # num_row = math.ceil(num_filters / num_column)
#         # print("num_row", num_row, "num_column", num_column)
#         # multi_frame = create_multi_frame(W_print, num_row, num_column, (256, 256))
#         # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         # cv2.imwrite("bvs/video/"+str(t)+"_W"+str(i)+"_inibition_filter.jpeg", heatmap.astype(np.uint8))
#         #
#         # J_print = np.swapaxes(J_print, 2, 3)
#         # multi_frame = create_multi_frame(J_print, num_row, num_column, (256, 256))
#         # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         # cv2.imwrite("bvs/video/"+str(t)+"_J"+str(i)+"_excitatory_filter.jpeg", heatmap.astype(np.uint8))
#         #
#         # # print inhib convolution
#         # print("shape inhib", np.shape(inhib))
#         # inhib = np.squeeze(inhib)
#         # print("min max inhib", np.min(inhib), np.max(inhib))
#         # inhib = np.array(inhib * 255).astype(np.uint8)
#         # heatmap = cv2.applyColorMap(inhib, cv2.COLORMAP_VIRIDIS)
#         # cv2.imwrite("bvs/video/"+str(t)+"_inhib" + str(i) + "_response.jpeg", heatmap.astype(np.uint8))
#         #
#         # # print excit convolution
#         # print("shape excit", np.shape(excit))
#         # excit = np.squeeze(excit)
#         # print("min max excit", np.min(excit), np.max(excit))
#         # excit = np.array(excit * 255).astype(np.uint8)
#         # heatmap = cv2.applyColorMap(excit, cv2.COLORMAP_VIRIDIS)
#         # cv2.imwrite("bvs/video/"+str(t)+"_excit" + str(i) + "_response.jpeg", heatmap.astype(np.uint8))
#
#     print("shape inhibs", np.shape(inhibs))
#     inhibs = np.swapaxes(np.expand_dims(np.squeeze(inhibs), axis=3), 3, 0)
#     excits = np.swapaxes(np.expand_dims(np.squeeze(excits), axis=3), 3, 0)
#
#     if save_intermediate_img:
#         # save i_norm
#         i_norm_print = np.expand_dims(i_norm[0], axis=2)
#         i_norm_print = np.array(i_norm_print).astype(np.uint8)
#         multi_frame = create_multi_frame(i_norm_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_03_i_norm.jpeg", heatmap.astype(np.uint8))
#         # save inhibition
#         inhibs_print = np.expand_dims(inhibs[0], axis=2)
#         inhibs_print = np.array(inhibs_print).astype(np.uint8)
#         num_filters = np.shape(inhibs)[-1]
#         num_column = min(num_filters, max_column)
#         num_row = math.ceil(num_filters / num_column)
#         multi_frame = create_multi_frame(inhibs_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_04_inibition_response.jpeg", heatmap.astype(np.uint8))
#         # save excitation
#         excits_print = np.expand_dims(excits[0], axis=2)
#         excits_print = np.array(excits_print).astype(np.uint8)
#         multi_frame = create_multi_frame(excits_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_05_exitatory_response.jpeg", heatmap.astype(np.uint8))
#
#     print("[convolution] 04")
#     print("[convolution] shape inhibs", np.shape(inhibs))
#     print("[convolution] min max inhibs", np.min(inhibs), np.max(inhibs))
#     print("[convolution] 05")
#     print("[convolution] shape excit", np.shape(excits))
#     print("[convolution] min max excit", np.min(excits), np.max(excits))
#     print()
#
#     # neural response
#     ####################################################################################################################
#     # y = -alphaY * y + gx(x) + inhibs + Ic + i_noise_y
#     y += epsilon * (-alphaY * y + gx(x) + inhibs + Ic + i_noise_y)
#     ####################################################################################################################
#     print("[Y response] 06")
#     print("[Y response] shape y", np.shape(y))
#     print("[Y response] min max y", np.min(y), np.max(y))
#     print("[Y response] 07")
#     print("[Y response] shape gy(y)", np.shape(gy(y)))
#     print("[Y response] min max gy(y)", np.min(gy(y)), np.max(gy(y)))
#     print()
#
#     if save_intermediate_img:
#         # print y neuronal response (inhibitory)
#         y_print = np.expand_dims(y[0], axis=2)
#         multi_frame = create_multi_frame(y_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_06_y_responses.jpeg", heatmap.astype(np.uint8))
#         # print gy(y)
#         gy_print = np.expand_dims(gy(y)[0], axis=2)
#         multi_frame = create_multi_frame(gy_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_07_gy(y)_response.jpeg", heatmap.astype(np.uint8))
#
#     # build inhibitory psi matrix
#     inhibs_psi = np.zeros(np.shape(y))
#     print("[psi matrix] 08")
#     print("[psi matrix] shape inhib_psi", np.shape(inhibs_psi))
#     for th in range(n_rot):
#         psi_tmp = []
#         for t_p in range(n_rot):
#             if th != t_p:
#                 theta = th * np.pi / n_rot
#                 theta_p = t_p * np.pi / n_rot
#                 a = np.abs(theta - theta_p)
#                 dth = min(a, np.pi - a)
#
#                 psi_tmp.append(psi(dth, n_rot) * gy(y[:, :, :, t_p]))
#         inhibs_psi[:, :, :, th] = np.sum(psi_tmp, axis=0)
#     print("[psi matrix] min man inhibs_psi", np.min(inhibs_psi), np.max(inhibs_psi))
#
#     # save inhib psi
#     inhibs_psi_print = np.expand_dims(inhibs_psi[0], axis=2)
#     multi_frame = create_multi_frame(inhibs_psi_print, num_row, num_column, (256, 256))
#     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#     cv2.imwrite("bvs/video/"+str(t)+"_08_inibition_psi.jpeg", heatmap.astype(np.uint8))
#
#     ####################################################################################################################
#     x_inhib = -alphaX * x - gy(y) - inhibs_psi
#     x_excit = J0 * gx(x) + excits + I_i_theta + i_norm + i_noise_x
#     # x = -alphaX * x - gy(y) - inhib_psi + J0 * gx(x) + excit  # term, I_{I, theta} and I0
#     # x = x_inhib + x_excit
#     x += epsilon * (x_inhib + x_excit)  # that what I understood from Zhaoping's li code
#     ####################################################################################################################
#     print("[X response] 09 min max x_inhib", np.min(x_inhib), np.max(x_inhib))
#     print("[X response] 10 min max x_excit", np.min(x_excit), np.max(x_excit))
#     print("[X response] 11 min max x", np.min(x), np.max(x))
#
#
#     if save_intermediate_img:
#         # plot V1 response
#         x_inhib_print = np.expand_dims(x_inhib[0], axis=2)
#         multi_frame = create_multi_frame(x_inhib_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_09_x_inhib_response.jpeg", heatmap.astype(np.uint8))
#
#         x_excit_print = np.expand_dims(x_excit[0], axis=2)
#         multi_frame = create_multi_frame(x_excit_print, num_row, num_column, (256, 256))
#         heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#         cv2.imwrite("bvs/video/"+str(t)+"_10_x_excit_response.jpeg", heatmap.astype(np.uint8))
#
#     x_print = np.expand_dims(x[0], axis=2)
#     # x_print = x_print - np.min(x_print)
#     # x_print = x_print / np.max(x_print)
#     # x_print[x_print < 0] = 0
#     print("min max x_print", np.min(x_print), np.max(x_print))
#     multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
#     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#     cv2.imwrite("bvs/video/"+str(t)+"_11_V1_x_response.jpeg", heatmap.astype(np.uint8))
#
#     x_print = np.expand_dims(gx(x)[0], axis=2)
#     multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
#     heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
#     cv2.imwrite("bvs/video/"+str(t)+"_12_V1_gx(x)_response.jpeg", heatmap.astype(np.uint8))
#
# #
# # saliency_map = np.expand_dims(np.squeeze(np.max(x, axis=3)), axis=2)
# # saliency_map = np.array(saliency_map * 255).astype(np.uint8)
# # print("shape saliency map", np.shape(saliency_map))
# # heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_VIRIDIS)
# # cv2.imwrite("bvs/video/V1_saliency_map.jpeg", heatmap.astype(np.uint8))
# #
#
#
