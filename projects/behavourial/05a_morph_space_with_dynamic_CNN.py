import os
from os.path import join
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from PIL import Image

from utils.load_config import *

from utils.load_config import load_config
from utils.load_data import load_data

from models.CNN.cornet_s import *
from models.CNN.MAE_DFER.build_model import create_MAE_DFER

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.05_morph_space_with_CNN
"""

#%% import config
config_path = 'BH_05_morph_space_with_CNN_MAE_DFER_w0001.json'

#%% declare script variables
# occluded and orignial are the same for this pipeline as we do not have any landmark on the ears
show_plot = True

config_paths = ["BH_05_morph_space_with_CNN_MAE_DFER_w0001.json"]
conditions = ["human_orig", "monkey_orig"]

config = load_config(config_path, path=r'C:\Users\Alex\Documents\Uni\NRE\BVS\configs\behavourial')

model = create_MAE_DFER().to('cpu')
model.fc_norm = None

def load_video_from_frames(directory):
    to_tensor = ToTensor()
    to160 = Resize(160)
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    T = len(os.listdir(directory))
    frames_to_keep = np.arange(0, T, step=(T / 16)).astype(int)
    x = torch.zeros((3, 16, 160, 160))

    imgs = []
    for img in os.listdir(directory):
        img = Image.open(join(directory, img))
        imgs.append(img)
    for i, idx in enumerate(frames_to_keep):
        img = imgs[idx]
        # img.show()
        img = to_tensor(img)
        img = to160(img)
        img = normalize(img)
        x[:, i, :, :] = img     # c, t, h, w
    x = torch.unsqueeze(x, 0)       # b, c, t, h, w
    return x


### Train
def train_morphing(model, config, condition):
    base_path = config['directory']
    if condition == 'human_orig':
        paths = [
            join(base_path, r'human_orig\HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0'),
            join(base_path, r'human_orig\HumanAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0'),
            join(base_path, r'human_orig\HumanAvatar_Anger_1.0_Fear_0.0_Monkey_0.0_Human_1.0'),
            join(base_path, r'human_orig\HumanAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0'),
        ]
    elif condition == 'monkey_orig':
        paths = [
            join(base_path, r'monkey_orig\MonkeyAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0'),
            join(base_path, r'monkey_orig\MonkeyAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0'),
            join(base_path, r'monkey_orig\MonkeyAvatar_Anger_1.0_Fear_0.0_Monkey_0.0_Human_1.0'),
            join(base_path, r'monkey_orig\MonkeyAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0'),
        ]
    print('norm:', model.norm)
    print('fc_norm:', model.fc_norm)
    tuning_vectors = torch.zeros((4, 512))
    for i, path in enumerate(paths):
        x = load_video_from_frames(path).to('cpu')
        print('x:', x.shape)
        out, z = model(x, save_feature=True)
        z = z.detach().cpu()
        tuning_vectors[i, :] = (z / torch.linalg.norm(z))
        print('Out distribution:', out)
    print('tuning vectors:', tuning_vectors @ tuning_vectors.T)
    return tuning_vectors

tuning_vectors = train_morphing(model, config, 'human_orig')

def predict(model, config, tuning_vectors, sample_path):
    x = load_video_from_frames(sample_path).to('cpu')
    print('x:', x.shape)
    z = model.forward_features(x).detach().cpu()
    z = z / torch.linalg.norm(z)
    preds = z @ tuning_vectors.T
    print(preds)
    return preds

base_path = config['directory']
paths = [
            join(base_path, r'human_orig\HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0'),
            join(base_path, r'human_orig\HumanAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0'),
            join(base_path, r'human_orig\HumanAvatar_Anger_1.0_Fear_0.0_Monkey_0.0_Human_1.0'),
            join(base_path, r'human_orig\HumanAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0'),
        ]

print('fc_norm:', model.fc_norm)
print('head:', model.head)

for sample_path in paths:
    y = predict(model, config, tuning_vectors, sample_path=sample_path)
    print('preds:', y)


for config_path in config_paths:
    for cond, condition in enumerate(conditions):

        base_path = join(config['directory'], condition)
        sample_paths = os.listdir(base_path)

        for sample_path in sample_paths:
            y = predict(model, config, tuning_vectors, join(base_path, sample_path))
            print(sample_path, y)

        # create directory
        save_path = os.path.join(config["directory"], "model_behav_preds")
        if not os.path.exists(save_path):
            os.mkdir(save_path)



        #%%
        def print_morph_space(amax_ms_grid=None, cat_grid=None, prob_grid=None,
                              title=None, show_plot=True, save=True, save_path=None):
            if amax_ms_grid is not None:
                fig, axs = plt.subplots(2, 2)
                pcm1 = axs[0, 0].imshow(amax_ms_grid[..., 0], cmap='viridis', interpolation='nearest')
                fig.colorbar(pcm1, ax=axs[0, 0])
                pcm2 = axs[0, 1].imshow(amax_ms_grid[..., 1], cmap='viridis', interpolation='nearest')
                fig.colorbar(pcm2, ax=axs[0, 1])
                pcm3 = axs[1, 0].imshow(amax_ms_grid[..., 2], cmap='viridis', interpolation='nearest')
                fig.colorbar(pcm3, ax=axs[1, 0])
                pcm4 = axs[1, 1].imshow(amax_ms_grid[..., 3], cmap='viridis', interpolation='nearest')
                fig.colorbar(pcm4, ax=axs[1, 1])

                if save:
                    if save_path is None:
                        plt.savefig(f"{title}_morph_space_read_out_values.jpeg")
                    else:
                        plt.savefig(os.path.join(save_path, f"{title}_morph_space_read_out_values.jpeg"))

            if cat_grid is not None:
                # print category grid
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].imshow(cat_grid[..., 0], cmap='hot', interpolation='nearest')
                axs[0, 1].imshow(cat_grid[..., 1], cmap='hot', interpolation='nearest')
                axs[1, 0].imshow(cat_grid[..., 2], cmap='hot', interpolation='nearest')
                axs[1, 1].imshow(cat_grid[..., 3], cmap='hot', interpolation='nearest')

                if save:
                    if save_path is None:
                        plt.savefig(f"{title}_morph_space_categories_values.jpeg")
                    else:
                        plt.savefig(os.path.join(save_path, f"{title}_morph_space_categories_values.jpeg"))

            # print probability grid
            if cat_grid is not None:
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].imshow(prob_grid[..., 0], cmap='viridis', interpolation='nearest')
                axs[0, 1].imshow(prob_grid[..., 1], cmap='viridis', interpolation='nearest')
                axs[1, 0].imshow(prob_grid[..., 2], cmap='viridis', interpolation='nearest')
                pcm = axs[1, 1].imshow(prob_grid[..., 3], cmap='viridis', interpolation='nearest')

                fig.colorbar(pcm, ax=axs[:, 1], shrink=0.7)

                if save:
                    if save_path is None:
                        plt.savefig(f"{title}_morph_space_probabilities_values.jpeg")
                    else:
                        plt.savefig(os.path.join(save_path, f"{title}_morph_space_probabilities_values.jpeg"))

                if show_plot:
                    plt.show()

        # rearrange into 25 videos of 150 frames (???)
        morph_space_data = np.reshape(preds, [25, 150, -1])
        print("shape morph_space_data", np.shape(morph_space_data))

        # get max values for each video and category
        amax_ms = np.amax(morph_space_data, axis=1)
        print("shape amax_ms", np.shape(amax_ms))
        print(amax_ms)

        # make into grid
        # rearrange into 5 categories x 5 repeats x 4 predicted categories
        amax_ms_grid = np.reshape(amax_ms, [5, 5, -1])
        amax_ms_grid = amax_ms_grid[..., 1:]  # remove neutral category
        print("shape amax_ms_grid", np.shape(amax_ms_grid))

        cat_grid = np.zeros((5, 5, 4))
        prob_grid = np.zeros((5, 5, 4))
        for i in range(np.shape(amax_ms_grid)[0]):
            for j in range(np.shape(amax_ms_grid)[0]):
                x = amax_ms_grid[i, j]
                print(i*5 + j, "x:", x, np.argmax(x))
                cat_grid[i, j, np.argmax(x)] = 1
                prob_grid[i, j] = np.exp(x) / sum(np.exp(x))

        print("test category plot")
        print(cat_grid[..., 2])
        print(cat_grid[..., 3])

        print("model saved in:", save_path)
        title = config['project'] + "_" + condition
        np.save(os.path.join(save_path, f"{title}_amax_ms_grid"), amax_ms_grid)
        np.save(os.path.join(save_path, f"{title}_cat_grid"), cat_grid)
        np.save(os.path.join(save_path, f"{title}_prob_grid"), prob_grid)

        # print morphing space
        print_morph_space(amax_ms_grid, cat_grid, prob_grid, show_plot=show_plot, title=title, save=True)
