import os
from os.path import join
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm
from torchvision import transforms

from utils.load_config import load_config
from utils.load_data import load_data
from utils.load_from_csv import *

from models.CNN.cornet_s import *

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.05_morph_space_with_CNN
"""

#%% import config
# config_path = 'BH_05_morph_space_with_CNN_VGG19_imagenet_w0001.json'              # OK
# config_path = 'BH_05_morph_space_with_CNN_VGG19_imagenet_conv33_w0001.json'       # OK
# config_path = 'BH_05_morph_space_with_CNN_VGG19_affectnet_w0001.json'             # OK
# config_path = 'BH_05_morph_space_with_CNN_ResNet50v2_imagenet_w0001.json'         # OK
# config_path = 'BH_05_morph_space_with_CNN_ResNet50v2_affectnet_w0001.json'        # OK
# config_path = 'BH_05_morph_space_with_CNN_CORNet_affectnet_w0001.json'            # OK
# load config

#%% declare script variables
# occluded and orignial are the same for this pipeline as we do not have any landmark on the ears
show_plot = True
# config_paths = ['BH_05_morph_space_with_CNN_VGG19_imagenet_w0001.json',
#                 'BH_05_morph_space_with_CNN_VGG19_imagenet_conv33_w0001.json',
#                 'BH_05_morph_space_with_CNN_VGG19_affectnet_w0001.json',
#                 'BH_05_morph_space_with_CNN_ResNet50v2_imagenet_w0001.json',
#                 'BH_05_morph_space_with_CNN_ResNet50v2_affectnet_w0001.json',
#                 'BH_05_morph_space_with_CNN_CORNet_affectnet_w0001.json']
config_paths = ["BH_05_morph_space_with_CNN_CORNet_imagenet_w0001.json"]
# config_paths = ["BH_05_morph_space_with_CNN_VGG19_imagenet_w0001.json"]
conditions = ["human_orig", "monkey_orig"]
load_path = r'C:\Users\Alex\Documents\Uni\NRE\Dataset\MorphingSpace\no_specific_training\cnn_tuning_vectors'

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict(config, orig):
    with open(join(load_path, config['project'] + '.pickle'), 'rb') as handle:
        tuning_vectors = pickle.load(handle)
    tuning_vectors = tuning_vectors[orig].cuda()

    if config["model_class"] == "cornet":
        model = CORnet_S()

    state_dict = torch.load(config["weights"])["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value

    model.load_state_dict(new_state_dict)
    model = remove_last_layer(model)
    model = model.to("cuda")
    model.eval()

    model = remove_last_layer(model)

    csv_path = join(config['directory'], 'morphing_space_' + orig + '.csv')
    length = get_dataset_length(csv_path)

    preds = torch.zeros((length, 4))

    print("Predicting...")
    with torch.no_grad():
        for i in tqdm(range(length)):
            img = load_PIL_from_csv(csv_path, i)
            print(img)
            img = transform(img).cuda()
            img = img.unsqueeze(0)
            yhat = model(img)
            print('yhat shape', yhat.shape)
            print('tuning vectors shape', tuning_vectors.shape)

            yhat = yhat / torch.linalg.norm(yhat)   # normalize
            yhat = yhat @ tuning_vectors.T
            print('yhat:', yhat)
            preds[i, :] = torch.nn.functional.softmax(yhat)
            print('preds:', preds[i, :])
            print()

    return preds.cpu().numpy()


for config_path in config_paths:
    config = load_config(config_path, path=r'C:\Users\Alex\Documents\Uni\NRE\BVS\configs\behavourial')
    for cond, condition in enumerate(conditions):

        morph_csv = [os.path.join(config['directory'], "morphing_space_human_orig.csv"),
                     os.path.join(config['directory'], "morphing_space_monkey_orig.csv")]

        # edit dictionary for single condition type
        if cond is not None:
            config["train_csv"] = morph_csv[cond]
            config["condition"] = condition
            if "human" in condition[cond]:
                config["avatar_types"] = ["human"]
            else:
                config["avatar_types"] = ["monkey"]

        # create directory
        save_path = os.path.join(config["directory"], config["load_directory"])
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        #%% load model
        preds = predict(config, condition)
        print("preds:", preds)
        print("shape preds", np.shape(preds))

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