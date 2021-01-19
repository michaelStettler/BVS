"""
2021/01/19
This script calculates the tuning vector for the human avatar and monkey avatar dataset and compares them.
"""

import os
import numpy as np

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase
from utils.plot_cnn_output import plot_cnn_output

# setting
retrain = False

# param, order: human, monkey
config_names = ["norm_base_compare_tuning_vector_t0001.json",
                "norm_base_compare_tuning_vector_t0002.json"]

# load config
configs = []
for config_name in config_names:
    configs.append(load_config(config_name))

# train models/load model
# fit reference and tuning vector
norm_base_list = []
for config in configs:
    try:
        if retrain:
            raise IOError("retrain = True")
        norm_base = NormBase(config, input_shape=(224, 224, 3), save_name=config["sub_folder"])
    except IOError:
        norm_base = NormBase(config, input_shape=(224,224,3))
        dataset = load_data(config, train=config["dataset"])
        norm_base.fit(dataset)
        norm_base.save_model(config, config["sub_folder"])
    norm_base_list.append(norm_base)

# extract vectors
ref_human = norm_base_list[0].r
tun1_human, tun2_human = norm_base_list[0].t_mean[[1,2]]
cat1_human = ref_human + tun1_human
cat2_human = ref_human + tun2_human

ref_monkey = norm_base_list[1].r
tun1_monkey, tun2_monkey = norm_base_list[1].t_mean[[1,2]]
cat1_monkey = ref_monkey + tun1_monkey
cat2_monkey = ref_monkey + tun2_monkey

# reshape vectors to (28,28,256)
for vector in [ref_human, tun1_human, tun2_human, cat1_human, cat2_human,
               ref_monkey,tun1_monkey,tun2_monkey,cat1_monkey,cat2_monkey]:
    vector.shape = (28,28,256)

# plot vectors
path = os.path.join("models/saved", configs[0]["save_name"])
if not os.path.exists(path): os.mkdir(path)

plot_cnn_output(ref_human ,path,"ref_human.png" , title="Average response to neutral expression, human avatar")
plot_cnn_output(tun1_human,path,"tun1_human.png", title="Tuning vector to expression 1, human avatar")
plot_cnn_output(cat1_human,path,"cat1_human.png", title="Average response to expression 1, human avatar")
plot_cnn_output(tun2_human,path,"tun2_human.png", title="Tuning vector to expression 2, human avatar")
plot_cnn_output(cat2_human,path,"cat2_human.png", title="Average response to expression 2, human avatar")

plot_cnn_output(ref_monkey ,path,"ref_monkey.png" , title="Average response to neutral expression, monkey avatar")
plot_cnn_output(tun1_monkey,path,"tun1_monkey.png", title="Tuning vector to expression 1, monkey avatar")
plot_cnn_output(cat1_monkey,path,"cat1_monkey.png", title="Average response to expression 1, monkey avatar")
plot_cnn_output(tun2_monkey,path,"tun2_monkey.png", title="Tuning vector to expression 2, monkey avatar")
plot_cnn_output(cat2_monkey,path,"cat2_monkey.png", title="Average response to expression 2, monkey avatar")