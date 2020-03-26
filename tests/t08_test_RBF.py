import numpy as np
import numpy.matlib
import time

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from bvs.layers import RBF

np.set_printoptions(precision=3, linewidth=120)

np.random.seed(0)  # for reproducibility
data = np.random.rand(1024, 40)  # 10 data points, 4 features
# data = np.random.rand(4, 3)  # 10 data points, 4 features
data = np.around(data, decimals=3)
print("data")
print(data)

start_vect = time.time()

# hard code RBF to have a result to compare
sigma = 0.1
dataA = np.matlib.repmat(data, np.shape(data)[0], 1).reshape((-1, np.shape(data)[0], np.shape(data)[1]))  # repetition of matrix
dataB = np.tile(data, np.shape(data)[0]).reshape((-1, np.shape(data)[0], np.shape(data)[1]))  # repetitions of rows
rbf2 = np.exp(-np.linalg.norm(dataA - dataB, axis=2)**2 / 2 / sigma ** 2)
print("rbf2")
print(rbf2)

dur_vect = time.time() - start_vect
print("duration", dur_vect)

start_vect = time.time()
dataA = tf.reshape(tf.tile(data, [tf.shape(data)[0], 1]), [-1, tf.shape(data)[0], tf.shape(data)[1]])
dataB = tf.reshape(tf.tile(data, [1, tf.shape(data)[0]]), [-1, tf.shape(data)[0], tf.shape(data)[1]])
rbf2 = tf.exp(-tf.norm(dataA - dataB, axis=2)**2 / 2 / sigma ** 2)
print("tf rbf2")
print(rbf2)

dur_vect = time.time() - start_vect
print("duration", dur_vect)
# ------------------------- test tf RBF layer ------------------------#
print()
print("---------------------------------------------------------------")
print("-----------------------  LAYER RBF  ---------------------------")
# todo take care of the batch issue! (input has to be a multiplier of 32)
#  But is RBF as a layer even a good idea ? -> doesn't compute over all possibilities
#  (don't return 1x1024x1024 but 32x32x32)
# -----------------   build model   ---------------
input = Input(shape=(40))
x = RBF(sigma)(input)
model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# print(model.summary())

# -----------------   predict model   ---------------
start_vect = time.time()
pred = model.predict(x=data)
print("pred")
print(pred)
dur_vect = time.time() - start_vect
print("duration", dur_vect)
