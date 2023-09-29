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

#%% declare weights and biases
import wandb

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
        'batch_size': {'value': 64},
        'l2': {'values': [0., 0.5, 0.7, 0.9]}
    }
}

pprint.pprint(sweep_config)

# create sweep id
project_name = config["project"]
sweep_id = wandb.sweep(sweep_config, entity="BVS", project=project_name)


#%%

def main():
    run = wandb.init(entity="BVS", project=project_name)

    # config = wandb.config

    train_data = load_data(config, get_raw=True)
    val_data = load_data(config, get_raw=True, train=False)

    # %%

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


    # print("-- Data loaded --")
    # print("n_train", n_train)
    # print("n_val", n_val)
    # print("n_steps", n_steps)
    # print()

    #%% create model
    model = CORnet_S()

    state_dict = torch.load(config["weights"])["state_dict"]
    # Remove incorrect prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value

    model.load_state_dict(new_state_dict)

    # replace linear layer
    model.decoder.linear = torch.nn.Linear(in_features=512, out_features=config["n_category"])

    # elif "CORNet" in config["project"]:
    #     preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    #     if load_custom_model:
    #         print("load custom CORNet model")
    #         base_model = tf.keras.models.load_model(weights)
    #         # remove last Dense and avg pooling
    #         base_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-4].output)



    model = model.to("cuda")
    model.train()

    ### For now, assume that the model should be trained from scratch

    # # apply transfer learning to base model
    # if config.get('transfer_layer'):
    #     fine_tune_at = 0
    #     for l, layer in enumerate(base_model.layers):
    #         if config['transfer_layer'] == layer.name:
    #             fine_tune_at = l
    #             print(l, "layer", layer.name)
    #             print("fine tune at:", fine_tune_at)
    #     for layer in base_model.layers[:fine_tune_at]:
    #         print(f"layer {layer} set to non-trainable")
    #         layer.trainable = False

    # declare layers for new model
    # declare layers

    # Define transforms and augmentation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=72)
    ])


    # apply l2 if present
    if config.get('l2_regularization'):
        lambda_l2 = wandb.config.l2
    else:
        lambda_l2 = 0

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.lr, weight_decay=lambda_l2)    # set initial lr
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=lambda_l2)    # set initial lr

    # loss
    cross_entropy = nn.CrossEntropyLoss()


    # set learning scheduler
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=wandb.config.decay_rate)
    # lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)


    best_acc = 0
    for epoch in range(wandb.config.epoch):
    # for epoch in range(5):
        epoch_loss = 0

        # lr decay
        if epoch % wandb.config.decay_step == 0 and epoch > 0:
            print("Adjusted lr at epoch", epoch)
            lr_schedule.step()

        # train
        model.train()
        for x, y in train_loader:
            x, y = x.float(), y.long()   # torch doesn't deal well with half precision, so convert
            x = data_augmentation(x)
            x, y = x.to("cuda"), y.to("cuda")
            print('x', torch.max(x))
            x = normalize(x)
            print('x', torch.max(x))

            yhat = model(x)

            loss = cross_entropy(yhat, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss

        if epoch % 1 == 0:
            print("Epoch:", epoch, "Loss:", epoch_loss)

        # eval
        model.eval()

        # train
        n_correct = 0
        for x, y in train_loader:
            with torch.no_grad():
                x, y = x.float(), y.long()
                x, y = x.to("cuda"), y.to("cuda")
                x = normalize(x)
                yhat = model(x)
                pred = torch.argmax(yhat, axis=1)

                n_correct += torch.sum(pred == y)
        train_acc = n_correct / len(train_dataset)

        n_correct = 0
        for x, y in val_loader:
            with torch.no_grad():
                x, y = x.float(), y.long()
                x, y = x.to("cuda"), y.to("cuda")
                x = normalize(x)
                yhat = model(x)
                pred = torch.argmax(yhat, axis=1)

                n_correct += torch.sum(pred == y)
        val_acc = n_correct / len(val_dataset)
        if epoch % 1 == 0:
            print("Train accuracy:", train_acc)
            print("Val accuracy:", val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            # save model
            save_path = os.path.join(config["directory"], "saved_model")
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, config["project"] + "_" + run.name))

    # Log epoch values
    wandb.log({
        'val_acc': val_acc
    })


# if __name__ == "__main__":
#     main()

wandb.agent(sweep_id, function=main)
