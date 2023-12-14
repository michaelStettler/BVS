import os
from os.path import join
import numpy as np
import datetime
import tensorflow as tf
import keras
import pprint
from tqdm import tqdm
import einops
import torch
import torchvision

from utils.load_config import load_config
from utils.load_data import load_data
from models.CNN.cornet_s import *

from sklearn.linear_model import LogisticRegression

np.random.seed(0)

"""
run: python -m projects.behavourial.03_CNN_training
tensorboard: tensorboard --logdir D:/PycharmProjects/BVS/logs/fit
"""

config_path = 'BH_03_CNN_training_CORNet_imagenet_w0001.json'
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
    preprocess = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    base_model = CORnet_S()
    state_dict = torch.load(config["weights"])["state_dict"]
    # Remove incorrect prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value

    base_model.load_state_dict(new_state_dict)
    base_model = remove_last_layer(base_model)

    base_model = base_model.to("cuda")
    base_model.eval()

else:
    raise Exception('No model specified.')



print(train_data)

x_train = (torch.tensor(train_data[0]) / 255.).float()
x_train = x_train.permute([0, 3, 1, 2])   # torch wants the channel dimension first
y_train = torch.tensor(train_data[1])

x_val = (torch.tensor(val_data[0]) / 255.).float()
x_val = x_val.permute([0, 3, 1, 2])  # torch wants the channel dimension first
y_val = torch.tensor(val_data[1])


train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


#%%
def train(base_model, dataloader):
    regression = LogisticRegression(max_iter=1000)
    features = []
    targets = []

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to('cuda')
            x = preprocess(x)
            z = base_model(x).detach().cpu()
            z = torch.flatten(z, start_dim=1).numpy()
            features.append(z)
            targets.append(target.cpu().numpy())

    flattened_features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    print('flattened shape:', flattened_features.shape)
    x_train = np.squeeze(flattened_features)
    y_train = targets
    print('ytrain:', y_train)

    print('Fitting Regression model...')
    regression.fit(x_train, y_train)
    return regression

def test(base_model, regression, dataloader):
    print('Testing base model...')
    features = []
    targets = []

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to('cuda')
            x = preprocess(x)
            z = base_model(x).detach().cpu()
            z = torch.flatten(z, start_dim=1).numpy()
            features.append(z)
            targets.append(target.cpu().numpy())

    flattened_features = np.concatenate(features, axis=0)
    y_test = np.concatenate(targets, axis=0)
    print('flattened shape:', flattened_features.shape)

    yhat = regression.predict(flattened_features)
    print('yhat:', yhat)
    print('Number of errors:', np.sum(yhat != y_test))
    return None

def test_full_model(model, dataloader):
    print('Testing...')
    preds = []
    targets = []

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to('cuda')
            x = preprocess(x)
            yhat = model(x)
            preds.append(yhat.cpu().numpy())
            targets.append(target)
    yhat = np.concatenate(preds)
    y = np.concatenate(targets)
    print('yhat shape', yhat.shape)
    print('yhat:', yhat)
    print('y shape', y.shape)
    yhat = np.argmax(yhat, axis=1)
    print('yhat:', yhat)
    print('Number of errors:', np.sum(yhat != y))
    return None


def add_readout(cnn, regression):
    print(regression.coef_.shape)
    print('Sum of regression weights:', np.sum(regression.coef_))
    flatten = torch.nn.Flatten()
    readout = torch.nn.Linear(regression.coef_.shape[1], regression.coef_.shape[0])
    softmax = torch.nn.Softmax()

    model = torch.nn.Sequential(cnn, flatten, readout, softmax)
    # Set weights
    print('Last layer:', model[-2])
    model[-2].weight = torch.nn.Parameter(torch.from_numpy(regression.coef_).float())
    model[-2].bias = torch.nn.Parameter(torch.from_numpy(regression.intercept_).float())
    print('Sum of linear layer weights:', torch.sum(model[-2].weight))

    return model.to('cuda')

#%%
regression = train(base_model, train_loader)

#%%
model = add_readout(base_model, regression)
test_full_model(model, val_loader)
test(base_model, regression, val_loader)
test(base_model, regression, val_loader)

save_path = join(save_path, config['project'])
if not os.path.exists(save_path):
    os.mkdir(save_path)

if config['project'] == 'CORNet_imagenet':
    print('Saving cornet...')
    torch.save(model.state_dict(), join(save_path, 'CORNet_imagenet'))
