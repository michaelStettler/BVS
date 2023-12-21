import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from tqdm import tqdm
from torchvision import transforms

from utils.load_config import load_config
from utils.load_data import load_data

from models.CNN.cornet_s import *

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.05_morph_space_with_CNN
"""

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# config_path = 'BH_05_morph_space_with_CNN_VGG19_imagenet_w0001.json'              # OK
# config_path = 'BH_05_morph_space_with_CNN_VGG19_imagenet_conv33_w0001.json'       # OK
# config_path = 'BH_05_morph_space_with_CNN_VGG19_affectnet_w0001.json'             # OK
# config_path = 'BH_05_morph_space_with_CNN_ResNet50v2_imagenet_w0001.json'         # OK
# config_path = 'BH_05_morph_space_with_CNN_ResNet50v2_affectnet_w0001.json'        # OK
config_path = 'BH_05_morph_space_with_CNN_CORNet_affectnet_w0001.json'            # OK
# load config

# occluded and orignial are the same for this pipeline as we do not have any landmark on the ears
show_plot = True
config_paths = ['BH_05_morph_space_with_CNN_VGG19_imagenet_w0001.json',
                'BH_05_morph_space_with_CNN_VGG19_imagenet_conv33_w0001.json',
                'BH_05_morph_space_with_CNN_VGG19_affectnet_w0001.json',
                'BH_05_morph_space_with_CNN_ResNet50v2_imagenet_w0001.json',
                'BH_05_morph_space_with_CNN_ResNet50v2_affectnet_w0001.json',
                'BH_05_morph_space_with_CNN_CORNet_affectnet_w0001.json']
# config_paths = ["BH_05_morph_space_with_CNN_CORNet_imagenet_w0001.json"]

config_path = "BH_05_morph_space_with_CNN_CORNet_affectnet_w0001.json"
conditions = ["human_orig", "monkey_orig"]


#%% FUNCTION DEFINITIONS

def build_cornet():
    cnn = CORnet_S()
    cnn = remove_last_layer(cnn)
    flatten = torch.nn.Flatten()
    readout = torch.nn.Linear(25088, 5)
    softmax = torch.nn.Softmax()

    model = torch.nn.Sequential(cnn, flatten, readout, softmax)
    return model


def predict_torch(config, morph_data):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if config["model_class"] == "cornet":
        model = build_cornet()
    print('Loading weights from:', os.path.join(config["load_directory"], config["project"]))
    state_dict = torch.load(os.path.join(config["load_directory"], config["project"], config['project']))
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    model.eval()

    
    ### Sanity check
    sanity_data = load_data(config, get_raw=True, train=False)

    x_sanity = (torch.tensor(sanity_data[0]) / 255.).float()
    x_sanity = rearrange(x_sanity, 'b h w c -> b c h w')
    y_sanity = torch.tensor(sanity_data[1])

    sanity_dataset = torch.utils.data.TensorDataset(x_sanity, y_sanity)

    sanity_loader = torch.utils.data.DataLoader(sanity_dataset, batch_size=config["batch_size"], shuffle=False)

    def test_full_model(model, dataloader):
        print('Performing Sanity check...')
        preds = []
        targets = []

        with torch.no_grad():
            for x, target in dataloader:
                x = x.to('cuda')
                x = normalize(x)
                yhat = model(x)
                preds.append(yhat.cpu().numpy())
                targets.append(target)
        yhat = np.concatenate(preds)
        y = np.concatenate(targets)
        print('yhat shape', yhat.shape)
        print('yhat:', yhat)
        print('y shape', y.shape)
        for i in range(len(y)):
            print(y[i], yhat[i])
        yhat = np.argmax(yhat, axis=1)
        print('yhat:', yhat)
        print('Number of errors:', np.sum(yhat != y))
        return None
    test_full_model(model, sanity_loader)
    ###

    X, Y = torch.tensor(morph_data[0]), torch.tensor(morph_data[1])
    print('shape X:', X.shape)

    X = X / 255.
    X = rearrange(X, 'b h w c -> b c h w')
    X = X.float().to("cuda")


    preds = torch.zeros((Y.shape[0], 5))

    print("Predicting...")
    with torch.no_grad():
        for i in tqdm(range(Y.shape[0])):
            sample = X[i, :, :, :]
            sample = rearrange(sample, 'c h w -> () c h w')
            sample = normalize(sample)
            yhat = model(sample)
            preds[i, :] = yhat

    return preds.cpu().numpy()



def load_morph_data(config, cond, condition):
    print(cond, condition)

    morph_csv = [os.path.join(config['directory'], "morphing_space_human_orig.csv"),
                 os.path.join(config['directory'], "morphing_space_monkey_orig.csv")]

    # edit dictionary for single condition type
    if cond is not None:
        config["train_csv"] = morph_csv[cond]
        config["condition"] = condition
        if "human" in condition:
            config["avatar_types"] = ["human"]
        elif 'monkey' in condition:
            config["avatar_types"] = ["monkey"]
        else:
            raise NameError('No avatar of that type.')
    print(config['train_csv'])
    print(config['condition'])
    print(config['avatar_types'])


    print('Loading data:', config['avatar_types'])
    morph_data = load_data(config, get_raw=True)
    print("-- Data loaded --")
    print("len train_data[0]", len(morph_data[0]))
    return morph_data


def make_predictions(config):
    if "ResNet" in config["project"]:
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    elif "VGG19" in config["project"]:
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
    elif "CORNet" in config["project"]:
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

    if "tensor_engine" in config:
        if config["tensor_engine"] == "torch":
            preds = predict_torch(config, morph_data)
    else:
        morph_data[0] = preprocess_input(morph_data[0])
        model = tf.keras.models.load_model(os.path.join(config["load_directory"], config["project"]))
        preds = model.predict(morph_data[0])
    print("preds:", preds)
    print("shape preds", np.shape(preds))
    return preds


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


def process_and_save_predictions(config, preds):
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
    prob_grid_soft = np.zeros((5, 5, 4))
    for i in range(np.shape(amax_ms_grid)[0]):
        for j in range(np.shape(amax_ms_grid)[0]):
            x = amax_ms_grid[i, j]
            print(i * 5 + j, "x:", x, np.argmax(x))

            ### Try Probgrid without softmax
            prob_grid[i, j] = x / np.sum(x)
            prob_grid_soft[i, j] = np.exp(x) / np.sum(np.exp(x))
            ###
            cat_grid[i, j, np.argmax(x)] = 1
            # prob_grid[i, j] = np.exp(x) / sum(np.exp(x))

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


#%% GET PREDS
pred_dict = {}
config = load_config(config_path, path=r'C:\Users\Alex\Documents\Uni\NRE\BVS\configs\behavourial')
print(config['model_name'])
project = config['project']
pred_dict[project] = {}

###
config['directory'] = 'C:/Users/Alex/Documents/Uni/NRE/Dataset/MorphingSpace'
config['load_directory'] = 'C:/Users/Alex/Documents/Uni/NRE/Dataset/MorphingSpace/DNN_weights/linear_fits'
###

# create directory
save_path = os.path.join(config["directory"], "model_behav_preds/linear_fits")
if not os.path.exists(save_path):
    os.mkdir(save_path)

for cond, condition in enumerate(conditions):
    morph_data = load_morph_data(config, cond, condition)

    preds = make_predictions(config)
    pred_dict[project][condition] = preds