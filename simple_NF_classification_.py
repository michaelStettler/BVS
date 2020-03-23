import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from bvs.layers import GaborFilters

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

print("Simple Neural Field Classification Model")

# load config
with open('config.json') as f:
  config = json.load(f)

# load data
img_path = '../data/02_FearGrin_1.0_120fps/02_FearGrin_1.0_120fps.0000.jpeg'
img = cv2.imread(img_path)

# crop and resize image
img = img[:, 280:1000, :]
img = cv2.resize(img, (256, 256))

# define parameters for gabor filter
n_rot = config['n_rot']
thetas = np.array(range(n_rot)) / n_rot * np.pi
sigmas = config['sigmas']
lamdas = config['lamdas'] * np.pi

# build model
input = Input(shape=(256, 256, 3))
gabor_layer = GaborFilters((15, 15), theta=thetas, sigma=sigmas, lamda=lamdas, per_channel=True)
x = gabor_layer(input)

model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

test_img = np.expand_dims(img, axis=0)
pred = model.predict(x=test_img)
print("shape pred", np.shape(pred))

# todo continue pipeline
# todo max pooling
# todo pca red
# todo rbf
# todo neural field

# todo visualize gabor on real image
# todo add arrows for orientations filter, frequencies, etc.


