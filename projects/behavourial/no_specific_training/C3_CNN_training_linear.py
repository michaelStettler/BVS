import os
from os.path import join
import numpy as np
import datetime
import tensorflow as tf
import torch
import torchvision.transforms.functional
from torch import nn
import pprint
from PIL import Image
import pickle

from torchvision import transforms


from utils.load_config import load_config
from utils.load_data import load_data
from datasets_utils.morphing_space import get_morph_extremes_idx

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

#%% declare weights and biases

save_path = r'C:\Users\Alex\Documents\Uni\NRE\Dataset\MorphingSpace\no_specific_training\cnn_tuning_vectors'
data_path = r'C:\Users\Alex\Documents\Uni\NRE\Dataset\MorphingSpace'

train_imgs = {'human_orig': ['HumanAvatar_Anger_1.0_Fear_0.0_Monkey_0.0_Human_1.0.0048.jpeg',
                             'HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0.0062.jpeg',
                             'HumanAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0.0052.jpg',
                             'HumanAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0.0051.jpg'],
              'monkey_orig': ['MonkeyAvatar_Anger_1.0_Fear_0.0_Monkey_0.0_Human_1.0.0048.jpeg',
                              'MonkeyAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0.0048.jpeg',
                              'MonkeyAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0.0055.jpeg',
                              'MonkeyAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0.0037.jpeg']
              }

def main():

    model = CORnet_S()
    print(model)

    state_dict = torch.load(config["weights"])["state_dict"]
    # Remove incorrect prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value

    model.load_state_dict(new_state_dict)
    model = remove_last_layer(model)

    model.train()

    # Define transforms and augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tuning_vectors = {}

    for origin in train_imgs.keys():
        origin_path = join(data_path, origin)
        for i, img in enumerate(train_imgs[origin]):
            dir = join(origin_path, img[:-9])
            try:
                img = Image.open(join(dir, img))
            except:
                img = Image.open(join(dir, img[:-1]))    # some images have .jpeg extension so remove one more letter
            img = transform(img)
            img = img.unsqueeze(0)   # add batch dimension

            y = model(img)
            if i == 0:
                t = torch.zeros((4, y.shape[-1]))
            t[i, :] = y / torch.linalg.norm(y)   # normalize tuning vector for cosine
        print(t.shape)
        tuning_vectors[origin] = t

        print(t @ t.T)


    print('Saving...')
    print(tuning_vectors)
    with open(join(save_path, config['project'] + '.pickle'), 'wb') as handle:
        pickle.dump(tuning_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

