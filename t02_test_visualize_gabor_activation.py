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

print("Visualize gabor filter")

# load config
# config = 'config.json'
config = 'config_test2.json'
with open(config) as f:
  config = json.load(f)

# load data
img_path = '../data/02_FearGrin_1.0_120fps/02_FearGrin_1.0_120fps.0000.jpeg'
img = cv2.imread(img_path)

# crop and resize image
img = img[:, 280:1000, :]
img = cv2.resize(img, (256, 256))

# # test case img
# img = np.zeros((256, 256, 3))
# img[:128, :128, 0] = 255
# img[128:, 128:, 1] = 255
# plt.imshow(img)

print("Num orientations {}, sigmas {}, gamma {}".format(config['n_rot'], config['sigmas'], config['gamma']))

# build model
input = Input(shape=(256, 256, 3))

n_rot = config['n_rot']
thetas = np.array(range(n_rot)) / n_rot * np.pi
sigmas = config['sigmas']
lamdas = np.array(config['lamdas']) * np.pi
gamma = config['gamma']
gabor_layer = GaborFilters((15, 15), theta=thetas, sigma=sigmas, lamda=lamdas, gamma=gamma, per_channel=False)
x = gabor_layer(input)
# print("shape gabor_kernel", np.shape(gabor_layer.kernel))

# g_kernels = np.moveaxis(gabor_layer.kernel, -1, 0)
# for gb in g_kernels:
#     if np.shape(gb)[-1] == 1:
#         gb = np.squeeze(gb)
#     gb = (gb+1)/2
#     plt.figure()
#     plt.imshow(gb.astype(np.float32))
# plt.show()

kernels = gabor_layer.kernel
num_kernels = np.shape(kernels)[-1]
print("shape kernels", np.shape(kernels))
num_column = min(num_kernels, 4)
num_row = math.ceil(num_kernels / 4)
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
num_column = min(num_activations, 4)
num_row = math.ceil(num_activations / 4)
print("num column", num_column, "num_row", num_row)
multi_frame = create_multi_frame_heatmap(img, activations, num_row, num_column, (256, 256))

# plt.figure()
# plt.imshow(multi_frame)
# plt.show()
cv2.imwrite("bvs/video/heatmap_gabor_filters.jpeg", multi_frame.astype(np.uint8))

# activations = np.moveaxis(pred[0], -1, 0)
# for filter in activations:
#     img = img - np.min(img)
#     img = np.array((img / np.max(img)) * 255).astype(np.uint8)
#     filter = (filter - np.min(filter))
#     filter = filter / np.max(filter)
#     filter = np.array(filter * 255).astype(np.uint8)
#
#     alpha = 0.2
#     # heatmap = cv2.applyColorMap(filter, cv2.COLORMAP_VIRIDIS)
#     heatmap = cv2.applyColorMap(filter, cv2.COLORMAP_HOT)
#     output = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
#
#     # cv2.imshow("test", output)
#     # cv2.waitKey(0)
#
#     plt.figure()
#     plt.imshow(output)

# plt.show()

