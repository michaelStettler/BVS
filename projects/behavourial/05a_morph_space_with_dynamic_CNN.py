import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from einops import rearrange
from torchvision.transforms import Resize

from utils.load_config import load_config

np.random.seed(0)

"""
run: python -m projects.behavourial.03b_CNN_training_torch
tensorboard: tensorboard --logdir D:/PycharmProjects/BVS/logs/fit
"""

from models.CNN.M3DFEL import M3DFEL
from models.CNN.former_dfer.ST_Former import GenerateModel, RecorderMeter

#%% import config

config_paths = ['BH_05_morph_space_with_dynamics_M3DFEL_w0001.json']
config_paths = ['BH_05_morph_space_with_dynamics_FORMER_DFER_w0001.json']

#%% declare script variables
# occluded and orignial are the same for this pipeline as we do not have any landmark on the ears
show_plot = True

conditions = ["human_orig", "monkey_orig"]

def load_video_from_frames(directory, transform, frames_to_keep):
    imgs = []
    for img in os.listdir(directory):
        img = Image.open(join(directory, img))
        img = torchvision.transforms.functional.to_tensor(img)
        imgs.append(img)
    video = [imgs[idx] for idx in frames_to_keep]
    x = torch.stack(video)
    x = transform(x)
    x = rearrange(x, "t c h w -> 1 t c h w")
    return x

def get_preprocessing_procedure(config):
    if config["project"] == "MAE_DFER":
        model = create_MAE_DFER().to('cpu')
        model.fc_norm = None

    elif config["project"] == "M3DFEL":
        model = M3DFEL()
        weight_path = r"D:\PycharmProjects\dynamic_face_models\TFace\attribute\M3DFEL\outputs\DFEW-[12_17]_[20_29]\model_best.pth"
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.remove_head()
        transform = Resize(112)
        frames_to_keep = np.linspace(0, 149, num=16).astype(int)
    elif config["project"] == "FORMER_DFER":
        model = GenerateModel()
        weight_path = config["weights"]
        checkpoint = torch.load(weight_path)
        new_statedict = {}
        for key, val in checkpoint["state_dict"].items():
            s = key.replace("module.", "")
            print(key, s)
            new_statedict[s] = val
        model.load_state_dict(new_statedict)
        model.remove_head()
        transform = Resize(112)
        frames_to_keep = np.linspace(0, 149, num=16).astype(int)

    model.eval()
    return model, transform, frames_to_keep




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
    tuning_vectors = []
    for i, path in enumerate(paths):
        x = load_video_from_frames(directory=path, transform=transform, frames_to_keep=frames_to_keep).to('cpu')
        z = model(x)
        z = z.detach().cpu()
        tuning_vectors.append(z / torch.linalg.norm(z))
    tuning_vectors = torch.stack(tuning_vectors).squeeze()
    print("tuning vectors shape:", tuning_vectors.shape)
    print('tuning vectors:', tuning_vectors @ tuning_vectors.T)
    return tuning_vectors

def predict(model, tuning_vectors, sample_path, transform, frames_to_keep):
    x = load_video_from_frames(sample_path, transform, frames_to_keep).to('cpu')
    z = model(x).detach().cpu()
    z = z / torch.linalg.norm(z)
    pred = z @ tuning_vectors.T
    pred = torch.exp(20 * pred) / torch.sum(np.exp(20 * pred))
    return pred

for config_path in config_paths:
    # load config
    config = load_config(config_path, path=r'D:\PycharmProjects\BVS\configs\behavourial')
    model, transform, frames_to_keep = get_preprocessing_procedure(config)

    project_name = config["project"]
    for cond, condition in enumerate(conditions):

        tuning_vectors = train_morphing(model, config, condition)

        base_path = join(config['directory'], condition)
        sample_paths = os.listdir(base_path)

        preds = []
        for sample_path in sample_paths:
            y = predict(model, tuning_vectors, join(base_path, sample_path), transform, frames_to_keep)
            preds.append(y.detach().cpu().numpy())
            print(sample_path, y)
        preds = np.stack(preds)

        # create directory
        save_path = os.path.join(config["directory"], "model_behav_preds/linear_fits")
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
        print("shape preds:", preds.shape)

        # rearrange into 5 categories x 5 repeats x 4 predicted categories
        amax_ms_grid = np.reshape(preds, [5, 5, -1])
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
