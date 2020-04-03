import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from bvs.layers import GaborFilters
from bvs.utils.create_preds_seq import create_multi_frame
from bvs.utils.create_preds_seq import create_multi_frame_heatmap

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
thetas = np.array(range(n_rot)) / n_rot * np.pi
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

# g_kernels = np.moveaxis(gabor_layer.kernel, -1, 0)
# for gb in g_kernels:
#     if np.shape(gb)[-1] == 1:
#         gb = np.squeeze(gb)
#     gb = (gb+1)/2
#     plt.figure()
#     plt.imshow(gb.astype(np.float32))
# plt.show()
# print("shape g_kernels", np.shape(g_kernels))
# print(g_kernels[0])

kernels = gabor_layer.kernel
num_kernels = np.shape(kernels)[-1]
print("shape kernels", np.shape(kernels))
num_column = min(num_kernels, n_rot)
num_row = math.ceil(num_kernels / num_column)
print("num column", num_column, "num_row", num_row)
multi_frame = create_multi_frame(kernels, num_row, num_column, (256, 256))
cv2.imwrite("bvs/video/gabor_filters.jpeg", multi_frame.astype(np.uint8))


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
num_column = min(num_activations, n_rot)
num_row = math.ceil(num_activations / num_column)
print("num column", num_column, "num_row", num_row)
multi_frame = create_multi_frame_heatmap(img, activations, num_row, num_column, (np.shape(img)[1], np.shape(img)[0]))

# plt.figure()
# plt.imshow(multi_frame)
# plt.show()
cv2.imwrite("bvs/video/heatmap_gabor_filters.jpeg", multi_frame.astype(np.uint8))

print("shape activations", np.shape(activations))
saliency_type = "max"
if saliency_type == 'max':
    saliency = activations.max(axis=2)
elif saliency_type == 'sum':
    saliency = np.sum(activations, axis=2)
elif saliency_type == "ReLu_sum":
    activations[activations < 0] = 0
    saliency = np.sum(activations, axis=2)
else:
    raise NotImplementedError("please chose a valid saliency type!")

# normalize saliency
saliency = saliency - np.min(saliency)
saliency = saliency / np.max(saliency)
saliency *= 255
print("shape saliency", np.shape(saliency))
cv2.imwrite("bvs/video/saliency.jpeg", saliency.astype(np.uint8))

# --------------------- Zhaoping li's V1 Saliency Model --------------------- #
# start from activation of each filter
# normalize outputs
activations = activations - np.min(activations)
activations = activations / np.max(activations)

# declare variables


def gx(x, Tx=1):
    if x <= 0:
        return 0
    elif x <= Tx:
        return x
    else:
        return 1


def gy(y, Ly=1.2, g1=0.21, g2=2.5):
    if y < 0:
        return 0
    elif y <= Ly:
        return g1 * y
    else:
        return g1 * Ly * g2 * (y - Ly)


def psi(theta, K):
    theta = np.abs(theta)
    if theta == 0:
        return 1
    elif theta == np.pi / K:
        return 0.8
    elif theta == 2 * np.pi / K:
        return 0.7
    else:
        return 0

# W = np.zeros((np.shape(img)[0], n_rot))
# W = np.zeros((11, 11))
# print("shape W", np.shape(W))
# for i in range(W.shape[0]):
# for i in range(10):
#     # for j in range(W.shape[0]):
#     for j in range(10):
#         d = np.sqrt(i**2 + j**2)
#         print("i, j, d", i, j, d, d == 0)
#         if d != 0:
#             for theta in range(W.shape[1]):
#                 for theta_p in range(W.shape[1]):
#                     beta = 2 * np.abs(theta*np.pi/n_rot) + 2 * np.sin(np.abs(theta*np.pi/n_rot + theta_p*np.pi/n_rot))
#                     d_theta = np.abs(theta - theta_p) * np.pi / n_rot
#                     if d < 10 / np.cos(beta/4) or beta > np.pi / 1.1:
#                         W[i, theta] = 0.141 * (1 - np.exp(-0.4 * np.power(beta/d, 1.5)))*np.exp(-np.power(d_theta/(np.pi/4), 1.5))
i = 0
i_range = 21
translate = int(i_range/2)
theta1 = 0

W = np.zeros((i_range, n_rot))
print("shape W", np.shape(W))
for j in range(i_range):
    for theta_p in range(n_rot):
        d = np.sqrt((i-(j-translate))**2)
        print("j", j, "theta_p", theta_p)
        beta = 2 * theta1 + 2 * np.sin(np.abs(theta1 + theta_p * np.pi / n_rot))
        d_theta = np.abs(theta1 - theta_p) * np.pi / n_rot
        print("d", d, "d_theta", d_theta)
        d_max = 10 * np.cos(beta/4)
        print("d_max", d_max)
        if d != 0 and d < d_max:
            W[j, theta_p] = 1

np.set_printoptions(precision=3)
print(W)



