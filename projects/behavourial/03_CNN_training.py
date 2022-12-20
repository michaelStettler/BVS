import os
import numpy as np
import datetime
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data

np.random.seed(0)

"""
run: python -m projects.behavourial.03_CNN_training
tensorboard: tensorboard --logdir D:/PycharmProjects/BVS/logs/fit
"""

#%% import config
config_path = 'BH_03_CNN_training_ResNet50v2_w0001.json'  # ResNet50v2_imagenet
config_path = 'BH_03_CNN_training_VGG19_w0001.json'       # VGG19_imagenet
# load config
config = load_config(config_path, path='configs/behavourial')

#%% declare weights and biases
import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger
run = wandb.init(project=config["project"], entity="bvs")

#%% import data
train_data = load_data(config, get_raw=True)
val_data = load_data(config, get_raw=True, train=False)
n_train = len(train_data[0])
n_val = len(val_data[0])
n_steps = n_train/config["batch_size"]
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))
train_dataset = train_dataset.shuffle(600).batch(config["batch_size"])
val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1]))
val_dataset = val_dataset.shuffle(600).batch(config["batch_size"])
print("-- Data loaded --")
print("n_train", n_train)
print("n_val", n_val)
print("n_steps", n_steps)
print()

#%% set class weight
class_weight = {}
for i in range(config["n_category"]):
    n_data = len(train_data[1][train_data[1] == i])
    class_weight[i] = (1/n_data) * (len(train_data[1]) / 4)
    print("num data {}: {}".format(i, n_data))
# class_weight = {0: 0.5, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
class_weight = None
print("class weights:", class_weight)

#%% create wandb config
wandb.config = {
    "learning_rate": config["base_learning_rate"],
    "epochs": config["initial_epochs"],
    "momentum": config["momentum"],
    "batch_size": config["batch_size"]
}

#%% create model
# declare layers
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])
if "ResNet" in config["project"]:
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=config["include_top"], weights=config["weights"])
elif "VGG19" in config["project"]:
    preprocess_input = tf.keras.applications.vgg19.preprocess_input
    base_model = tf.keras.applications.vgg19.VGG19(include_top=config["include_top"], weights=config["weights"])
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(config["n_category"], activation='relu')

# apply transfer learning to base model
inputs = tf.keras.Input(shape=config["input_shape"])
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config["base_learning_rate"],
                                                             decay_steps=config["decay_steps"]*n_steps,
                                                             decay_rate=config['decay_rate'],
                                                             staircase=True)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config["momentum"]),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.summary()

#%% Train de model
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_dataset,
          epochs=config["initial_epochs"],
          batch_size=config["batch_size"],
          class_weight=class_weight,
          validation_data=val_dataset,
          callbacks=[tensorboard_callback, WandbMetricsLogger()])
callbacks = [WandbCallback()]

save_path = os.path.join(config["directory"], "saved_model")
if not os.path.isdir(save_path):
    os.mkdir(save_path)
run.save()  # nned to call this before being able to get the name
model.save(os.path.join(save_path, config["extraction_model"] + "_" + wandb.run.name))
