import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from bvs.layers import GaborFilters
from bvs.layers import MaxPoolDepths
from bvs.utils.create_preds_seq import create_multi_frame_heatmap

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

print("Visualize gabor filter")

# load config
# config = 'configs/config.json'
# config = 'configs/config_test3.json'
config = 'configs/config_test3_edge.json'
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

# build model
input = Input(shape=(256, 256, 3))

# -------------------- Gabor 1 ------------------ #
conf = config['Gabor'][0]
n_rot = conf['n_rot']
thetas = np.array(range(n_rot)) / n_rot * np.pi
gabor_layer1 = GaborFilters((15, 15), theta=thetas,
                            sigma=conf['sigmas'],
                            lamda=np.array(conf['lamdas']) * np.pi,
                            gamma=conf['gamma'],
                            phi=np.array(conf['phi']) * np.pi,
                            per_channel=False)
x1 = gabor_layer1(input)

# -------------------- Gabor 2 ------------------ #
conf = config['Gabor'][1]
n_rot = conf['n_rot']
thetas = np.array(range(n_rot)) / n_rot * np.pi
gabor_layer2 = GaborFilters((15, 15), theta=thetas,
                            sigma=conf['sigmas'],
                            lamda=np.array(conf['lamdas']) * np.pi,
                            gamma=conf['gamma'],
                            phi=np.array(conf['phi']) * np.pi,
                            per_channel=False)
x2 = gabor_layer2(input)
# -------------------- Gabor 3 ------------------ #
conf = config['Gabor'][2]
n_rot = conf['n_rot']
thetas = np.array(range(n_rot)) / n_rot * np.pi
gabor_layer3 = GaborFilters((15, 15), theta=thetas,
                            sigma=conf['sigmas'],
                            lamda=np.array(conf['lamdas']) * np.pi,
                            gamma=conf['gamma'],
                            phi=np.array(conf['phi']) * np.pi,
                            per_channel=False)
x3 = gabor_layer3(input)


# print("shape gabor_kernel", np.shape(gabor_layer.kernel))
print("shape layers", np.shape(x1), np.shape(x2), np.shape(x3))
x = tf.concat([x1, x2, x3], axis=3)
print("shape x", np.shape(x))

# pooling_layer = MaxPool2D(pool_size=(3, 3), strides=1, padding='SAME')
pooling_layer = MaxPoolDepths(ksize=(3, 3), strides=1, padding='SAME', axis=3, num_cond=3)
x = pooling_layer(x)

# g_kernels = np.moveaxis(gabor_layer.kernel, -1, 0)
# for gb in g_kernels:
#     if np.shape(gb)[-1] == 1:
#         gb = np.squeeze(gb)
#     gb = (gb+1)/2
#     plt.figure()
#     plt.imshow(gb.astype(np.float32))
# plt.show()

model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

test_img = np.expand_dims(img, axis=0)
pred = model.predict(x=test_img)
print("shape pred", np.shape(pred))

activations = pred[0]
num_activations = np.shape(activations)[-1]
num_column = min(num_activations, 4)
num_row = math.ceil(num_activations / 4)
print("num column", num_column, "num_row", num_row)
multi_frame = create_multi_frame_heatmap(img, activations, num_row, num_column, (256, 256))
cv2.imwrite("bvs/video/heatmap_MaxPooling.jpeg", multi_frame.astype(np.uint8))


# img0 = pred[0]
# filters = np.moveaxis(img0, -1, 0)
# print("shape predictions", np.shape(pred))
# for filter in filters:
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
#
# plt.show()

