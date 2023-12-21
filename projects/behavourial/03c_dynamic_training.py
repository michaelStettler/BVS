import os
os.environ["WANDB_SILENT"] = "true"
import numpy as np
import datetime
import tensorflow as tf
import torch
from torch import nn
import pprint

from torchvision import transforms

from utils.load_config import load_config
from utils.load_data import load_data

from models.CNN.cornet_s import *

np.random.seed(0)

"""
run: python -m projects.behavourial.03b_CNN_training_torch
tensorboard: tensorboard --logdir D:/PycharmProjects/BVS/logs/fit
"""

#%% import config

config_path = 'BH_03_CNN_training_CORNet_imagenet_w0001.json'
# load config
config = load_config(config_path, path=r'C:\Users\Alex\Documents\Uni\NRE\BVS\configs\behavourial')
print(config)

project_name = config["project"]


#%%

train_data = load_data(config, get_raw=True)
val_data = load_data(config, get_raw=True, train=False)

#%%

x_train = torch.tensor(train_data[0]) / 255.
x_train = x_train.permute([0, 3, 1, 2])   # torch wants the channel dimension first
y_train = torch.tensor(train_data[1])

x_val = torch.tensor(val_data[0]) / 255.
x_val = x_val.permute([0, 3, 1, 2])  # torch wants the channel dimension first
y_val = torch.tensor(val_data[1])

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)


print("-- Data loaded --")
print("n_train", n_train)
print("n_val", n_val)
print("n_steps", n_steps)
print()

#%% create model
if project_name == 'test_name':
    model = ...

    # Define model-speficic transforms etc.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])




