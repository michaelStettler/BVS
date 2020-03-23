import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from bvs.layers import GaborFilters
from bvs.utils.create_preds_seq import create_preds_seq
from bvs.utils.unflatten_idx import unflatten_idx
import os
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

print("Visualize PCA activation")

# load config
# config = 'config.json'
config = 'config_test2.json'
with open(config) as f:
  config = json.load(f)

# load data
dir_path = '../data/02_FearGrin_1.0_120fps/'
list_img = sorted(os.listdir(dir_path))
data = []
for img_name in list_img:
    img = cv2.imread(dir_path + img_name)

    # crop and resize image
    img = img[:, 280:1000, :]
    img = cv2.resize(img, (256, 256))
    data.append(img)
data = np.array(data)

# # test case img
# img = np.zeros((256, 256, 3))
# img[:128, :128, 0] = 255
# img[128:, 128:, 1] = 255
# plt.imshow(img)

# -----------------   build model   ---------------
# define model parameters
n_rot = 8
thetas = np.array(range(n_rot)) / n_rot * np.pi
sigmas = config['sigmas']
lamdas = np.array(config['lamdas']) * np.pi
gamma = config['gamma']
maxPool_stride = 3

# construt layers
input = Input(shape=(256, 256, 3))
gabor_layer = GaborFilters((15, 15), theta=thetas, sigma=sigmas, lamda=lamdas, gamma=gamma, per_channel=False)
x = gabor_layer(input)
pooling_layer = MaxPool2D(pool_size=(3, 3), strides=maxPool_stride, padding='SAME')
x = pooling_layer(x)
v2 = GaborFilters((15, 15), theta=thetas, sigma=sigmas, lamda=lamdas, gamma=gamma, per_channel=True)
x = v2(x)
pooling_layer_v2 = MaxPool2D(pool_size=(3, 3), strides=maxPool_stride, padding='SAME')
x = pooling_layer_v2(x)
print("shape gabor_kernel", np.shape(gabor_layer.kernel))

model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

print("shape data", np.shape(data))
pred = model.predict(x=data)
print("shape pred", np.shape(pred))

# create_preds_seq(data, pred, max_column=8)

# ---------------------- PCA --------------------
# get PCA components
pca = PCA(n_components=50)
flatten_pred = np.reshape(pred, (np.shape(pred)[0], -1))
print("shape flatten_pred", np.shape(flatten_pred))
flatten_pred = flatten_pred - np.min(flatten_pred)
flatten_pred = flatten_pred / np.max(flatten_pred)
pca.fit(flatten_pred)
print("explained_variance_ratio_")
print(pca.explained_variance_ratio_)
n_pcs = pca.components_.shape[0]
# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
# flatten to 3d space
unflatten_idx = unflatten_idx(most_important, np.shape(pred)[1:])
print("shape unflatten_idx", np.shape(unflatten_idx))

for i, uidx in enumerate(unflatten_idx):
    print("components {}: {}".format(i, uidx))

# create heatmap according to the PCA most important feature
heatmap = np.zeros(np.shape(pred)[:-1])
print("shape heatmap", np.shape(heatmap))
for i, p in enumerate(pred):
    for u_idx in unflatten_idx:
        heatmap[i, u_idx[0], u_idx[1]] = p[u_idx]

# --------------------- PLOT ------------------------
# plot heatmap activation on sequence
# normalize the heatmap and images
images = data - np.min(data)
images = data / np.max(data)
heatmap = heatmap - np.min(heatmap)
heatmap = heatmap / np.max(heatmap)
# loop over each frame to save as image sequence
alpha = 0.2
for f in range(np.shape(data)[0]):
    # get orig image
    img = images[f]
    img = np.array(img * 255).astype(np.uint8)

    # get heatmap and resize to match orig image
    hm = heatmap[f]
    hm = np.array(hm * 255).astype(np.uint8)
    hm = cv2.resize(hm, (256, 256))

    # add heatmap to original image
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    output = cv2.addWeighted(img, alpha, hm, 1 - alpha, 0)

    # save image
    cv2.imwrite("bvs/video/seq_" + str(f) + ".jpeg", output.astype(np.uint8))


