import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from bvs.layers import GaborFilters
from bvs.layers import MaxPoolDepths
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
config = 'config_test3.json'
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
                            per_channel=False)
x3 = gabor_layer3(input)


# print("shape gabor_kernel", np.shape(gabor_layer.kernel))
print("shape layers", np.shape(x1), np.shape(x2), np.shape(x3))
x = tf.concat([x1, x2, x3], axis=3)
print("shape x", np.shape(x))

# pooling_layer = MaxPool2D(pool_size=(3, 3), strides=1, padding='SAME')
pooling_layer = MaxPoolDepths(ksize=(3, 3), strides=1, padding='SAME', axis=3, num_cond=3)
x = pooling_layer(x)

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


