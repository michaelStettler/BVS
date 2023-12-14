import os
from os.path import join
import numpy as np
import datetime
import tensorflow as tf
import keras
import pprint
from tqdm import tqdm
import einops

from utils.load_config import load_config
from utils.load_data import load_data

from sklearn.linear_model import LogisticRegression

np.random.seed(0)

"""
run: python -m projects.behavourial.03_CNN_training
tensorboard: tensorboard --logdir D:/PycharmProjects/BVS/logs/fit
"""

# config_path = 'BH_03_CNN_training_ResNet50v2_imagenet_w0001.json'           # ResNet50v2_imagenet DONE
# config_path = 'BH_03_CNN_training_ResNet50v2_affectnet_w0001.json'        # ResNet50v2_affectnet DONE
# config_path = 'BH_03_CNN_training_VGG19_imagnet_w0001.json'                # VGG19_imagenet DONE
# config_path = 'BH_03_CNN_training_VGG19_affectnet_w0001.json'              # VGG19_affectnet
# config_path = 'BH_03_CNN_training_VGG19_imagenet_conv33_w0001.json'         # VGG19_imagenet_conv3_3 DONE
#config_path = 'BH_03_CNN_training_VGG19_scratch_w0001.json'                # VGG19_imagenet_scratch
config_path = 'BH_03_CNN_training_CORNet_affectnet_w0001.json'            # CORNet_affectnet DONE
# load config
config = load_config(config_path, path=r'C:\Users\Alex\Documents\Uni\NRE\BVS\configs/behavourial')

# Fix path to Morphing space
for key in list(config.keys()):
    if isinstance(config[key], str):
        config[key] = config[key].replace('D:/Dataset', r'C:\Users\Alex\Documents\Uni\NRE\Dataset')
        config[key] = config[key].replace('D:/DNN_weights', r'C:\Users\Alex\Documents\Uni\NRE\Dataset\MorphingSpace\DNN_weights')

print('Loading data...')
train_data = load_data(config, get_raw=True)
val_data = load_data(config, get_raw=True, train=False)
n_train = len(train_data[0])
n_val = len(val_data[0])
print("-- Data loaded --")
print("n_train", n_train)
print("n_val", n_val)
print()

save_path = r'C:\Users\Alex\Documents\Uni\NRE\Dataset\MorphingSpace\DNN_weights\linear_fits'

# ### Debugging set
# train_data = [train_data[0][:50], train_data[1][:50]]
# val_data = [val_data[0][::4], val_data[1][::4]]



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
        base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
    else:
        base_model = tf.keras.applications.vgg19.VGG19(include_top=config["include_top"], weights=weights)
        print(base_model.layers)

elif "CORNet" in config["project"]:
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    if load_custom_model:
        print("load custom CORNet model")
        base_model = tf.keras.models.load_model(weights)
        # remove last Dense and avg pooling
        print(base_model.layers)
        # base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-4].output)
else:
    raise Exception('No model specified.')

def preprocess(input):
    '''
    Wrapper for preprocessing that copies the input first.
    '''
    return preprocess_input(np.copy(input))

class Preprocess(tf.keras.layers.Layer):
    '''
    Wrapper for preprocessing that copies the input first.
    '''
    def __init__(self):
        super(Preprocess, self).__init__(name=None)
        self.function = preprocess_input

    def call(self, inputs):
        out = self.function(tf.identity(inputs))
        print(tf.math.reduce_sum(out))
        return out



# Preprocess the data once before running training and inference
train_data[0] = preprocess(train_data[0])
val_data[0] = preprocess(val_data[0])

#%%

# print(val_data[0][10][:, :, 0].shape)
# print(val_data[0][10][:, :, 0])
#
# print(np.sum(val_data[0][10]))
# raise ValueError('debug')


def train(base_model, data):
    regression = LogisticRegression(max_iter=1000)

    features = base_model.predict(data[0])
    print('train features shape:', features.shape)
    dim = np.prod(features[0].shape)
    flatten = keras.layers.Reshape((1, dim))
    flattened_features = flatten(features)
    print('flattened features shape:', flattened_features.shape)

    x_train = np.squeeze(flattened_features)
    y_train = data[1]

    print('Fitting Regression model...')
    regression.fit(x_train, y_train)
    return regression

def test(base_model, regression, data):
    print('Entering test function...')

    features = base_model.predict(data[0])
    print('train features shape:', features.shape)
    dim = np.prod(features[0].shape)
    flatten = keras.layers.Reshape((1, dim))
    flattened_features = flatten(features)
    print('flattened features shape:', flattened_features.shape)

    x_test = np.squeeze(flattened_features)
    y_test = data[1]

    yhat = regression.predict(x_test)
    print('yhat:', yhat)
    print('Number of errors:', np.sum(yhat != y_test))
    return None

def test_full_model(model, data):
    y_test = data[1]

    print('Using predict method:')
    yhat = model.predict(data[0])
    yhat = np.squeeze(yhat)
    print('yhat shape:', yhat.shape)
    print('Output of softmax:')
    print(yhat)
    yhat = np.argmax(yhat, axis=1)
    print('yhat:', yhat)
    print('Number of errors:', np.sum(yhat != y_test))
    return None


def add_readout(cnn, regression):
    # Get shape of the cnn output layer
    x = tf.expand_dims(train_data[0][0], 0)
    dim_cnn = np.prod(cnn(x).shape)

    flatten = keras.layers.Reshape((1, dim_cnn))
    keras_readout = keras.layers.Dense(5, use_bias=True, activation=keras.activations.softmax, name='readout')
    model = keras.Sequential([cnn, flatten, keras_readout])

    model.get_layer('readout').set_weights([regression.coef_.T, regression.intercept_])

    return model

regression = train(base_model, train_data)
model = add_readout(base_model, regression)
test_full_model(model, val_data)
test(base_model, regression, val_data)

save_path = join(save_path, config['project'])
model.save(save_path)



