import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model

"""
run: python -m projects.behavourial.03_CNN_training
"""

#%% import config
config_path = 'BH_03_CNN_training_ResNet50v2_m0001.json'
# load config
config = load_config(config_path, path='configs/behavourial')

#%% import data
train_data = load_data(config)
print("-- Data loaded --")
print("len train_data[0]", len(train_data[0]))
print()

#%% create model
# declare layers
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=config["include_top"], weights=config["weights"])
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(config["n_category"])

# apply transfer learning to base model
inputs = tf.keras.Input(shape=config["input_shape"])
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=config["base_learning_rate"],
                                                             momentum=config["momentum"]),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#%% Train de model
history = model.fit(x=train_data[0], y=tf.one_hot(train_data[1], config["n_category"]),
                    epochs=config["initial_epochs"])