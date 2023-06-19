import os
os.environ["WANDB_SILENT"] = "true"
import numpy as np
import datetime
import tensorflow as tf
import pprint

from utils.load_config import load_config
from utils.load_data import load_data

np.random.seed(0)

"""
run: python -m projects.behavourial.03_CNN_training
tensorboard: tensorboard --logdir D:/PycharmProjects/BVS/logs/fit
"""

#%% import config
# config_path = 'BH_03_CNN_training_ResNet50v2_imagenet_w0001.json'           # ResNet50v2_imagenet DONE
# config_path = 'BH_03_CNN_training_ResNet50v2_affectnet_w0001.json'        # ResNet50v2_affectnet DONE
# config_path = 'BH_03_CNN_training_VGG19_imagnet_w0001.json'                # VGG19_imagenet DONE
config_path = 'BH_03_CNN_training_VGG19_affectnet_w0001.json'              # VGG19_affectnet
# config_path = 'BH_03_CNN_training_VGG19_imagenet_conv33_w0001.json'         # VGG19_imagenet_conv3_3 DONE
#config_path = 'BH_03_CNN_training_VGG19_scratch_w0001.json'                # VGG19_imagenet_scratch
# config_path = 'BH_03_CNN_training_CORNet_affectnet_w0001.json'            # CORNet_affectnet DONE
# load config
config = load_config(config_path, path='configs/behavourial')

#%% declare weights and biases
import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger

# Used for all but VGG AffectNet
sweep_config = {
    'method': 'grid',
    'metric': {
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
    },
    'parameters': {
        'lr': {'values': [0.02, 0.01, 0.001]},
        'epoch': {'values': [180]},
        'decay_step': {'values': [25, 30]},
        'decay_rate': {'values': [0., .9, .8, .7, .75]},
        'momentum': {'value': 0.9},
        'batch_size': {'value': 32},
        'l2': {'values': [0., 0.5, 0.7, 0.9]}
    }
}

# Used for VGG affectnet
# sweep_config = {
#     'method': 'grid',
#     'metric': {
#         'metric': {'goal': 'maximize', 'name': 'val_acc'},
#     },
#     'parameters': {
#         'lr': {'values': [0.001]},
#         'epoch': {'values': [180]},
#         'decay_step': {'values': [200]},
#         'decay_rate': {'values': [0.9]},
#         'momentum': {'value': 0.9},
#         'batch_size': {'value': 64},
#         'l2': {'values': [0.0]}
#     }
# }
pprint.pprint(sweep_config)

# create sweep id
project_name = config["project"]
sweep_id = wandb.sweep(sweep_config, entity="BVS", project=project_name)


def main():
    run = wandb.init(entity="BVS", project=project_name)
    # config = wandb.config

    train_data = load_data(config, get_raw=True)
    val_data = load_data(config, get_raw=True, train=False)
    n_train = len(train_data[0])
    n_val = len(val_data[0])
    n_steps = n_train / wandb.config.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1]))
    train_dataset = train_dataset.shuffle(600, reshuffle_each_iteration=True).batch(wandb.config.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1]))
    val_dataset = val_dataset.shuffle(600, reshuffle_each_iteration=True).batch(wandb.config.batch_size)
    print("-- Data loaded --")
    print("n_train", n_train)
    print("n_val", n_val)
    print("n_steps", n_steps)
    print()

    #%% create model

    # load weights
    load_custom_model = False
    if config["weights"] == "imagenet":
        weights = "imagenet"
    elif config["weights"] == "None":
        weights = None
    else:
        load_custom_model = True
        weights = config["weights"]
    print(f"Weight Loaded: {config['weights']}")

    if "ResNet" in config["project"]:
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        if load_custom_model:
            print("load custom ResNet model")
            base_model = tf.keras.models.load_model(weights)
            # remove last Dense and avg pooling
            base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
        else:
            base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=config["include_top"], weights=weights)

    elif "VGG19" in config["project"]:
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        if load_custom_model:
            print("load custom VGG model")
            base_model = tf.keras.models.load_model(weights)
            # base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-5].output)
            base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
        else:
            base_model = tf.keras.applications.vgg19.VGG19(include_top=config["include_top"], weights=weights)
            base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    elif "CORNet" in config["project"]:
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        if load_custom_model:
            print("load custom CORNet model")
            base_model = tf.keras.models.load_model(weights)
            # remove last Dense and avg pooling
            base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-4].output)


    base_model.training = True
    print(base_model.summary())
    print("end base model")

    # apply transfer learning to base model
    if config.get('transfer_layer'):
        fine_tune_at = 0
        for l, layer in enumerate(base_model.layers):
            if config['transfer_layer'] == layer.name:
                fine_tune_at = l
                print(l, "layer", layer.name)
                print("fine tune at:", fine_tune_at)
        for layer in base_model.layers[:fine_tune_at]:
            print(f"layer {layer} set to non-trainable")
            layer.trainable = False

    # declare layers for new model
    # declare layers
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(config["n_category"], activation='relu')

    # construct new model
    inputs = tf.keras.Input(shape=config["input_shape"])
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x)
    if "VGG" not in config["project"]:
        x = global_average_layer(x)
    # add a fully connected layer for VGGto correlate with original implementation
    if "VGG" in config["project"] and not config["include_top"] and not load_custom_model:
        print("add extra dense layer for VGG")  # only to train on affectnet
        fc = tf.keras.layers.Dense(512, activation='relu')
        x = fc(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # apply l2 if present
    if config.get('l2_regularization'):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_regularizer = tf.keras.regularizers.l2(wandb.config.l2)

    # set learning scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(wandb.config.lr,
                                                                 decay_steps=wandb.config.decay_step*n_steps,
                                                                 decay_rate=wandb.config.decay_rate,
                                                                 staircase=True)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=wandb.config.momentum),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.summary()
    print("end new model")

    #%% Train de model
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # set early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                               patience=wandb.config.epoch,
                                                               restore_best_weights=True)

    model.fit(train_dataset,
              epochs=wandb.config.epoch,
              batch_size=wandb.config.batch_size,
              validation_data=val_dataset,
              callbacks=[early_stopping_callback, tensorboard_callback, WandbMetricsLogger()])
    callbacks = [WandbCallback()]

    save_path = os.path.join(config["directory"], "saved_model")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # model.save(os.path.join(save_path, config["extraction_model"] + "_" + wandb.run.name))
    model.save(os.path.join(save_path, config["project"] + "_" + run.name))


wandb.agent(sweep_id, function=main)
