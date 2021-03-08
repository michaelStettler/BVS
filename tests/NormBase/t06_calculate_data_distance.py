"""
2021/01/07
This script figures out why the transfer didn't work but the intermediate model worked.
It trains the human and the monkey model separately and then calculates the combined model as the average.
Several data is printed out to the command line.
Important results:
- distance between human and monkey data set is high compared to in between distance
- tuning vectors of human and monkey data set are orthogonal.
- tuning vector of intermediate model is 45 degree (consequence of previous point)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.load_config import load_config
from models.NormBase import NormBase
from utils.load_data import load_data

config_names = ["norm_base_data_distance_t0001.json", "norm_base_data_distance_t0002.json"]
models = []

for i, config_name in enumerate(config_names):
    config = load_config(config_name)
    save_name = config["sub_folder"]
    retrain = False

    # model
    try:
        if retrain:
            raise IOError("retrain = True")
        norm_base = NormBase(config, input_shape=(224, 224, 3), save_name=save_name)
    except IOError:
        norm_base = NormBase(config, input_shape=(224,224,3))

        norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][0]), fit_dim_red=True, fit_ref=False,
                      fit_tun=False)
        norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][1]), fit_dim_red=False, fit_ref=True,
                      fit_tun=False)
        norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][2]), fit_dim_red=False, fit_ref=False,
                      fit_tun=True)
        norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][3]), fit_dim_red=False, fit_ref=True,
                      fit_tun=False)
        norm_base.save_model(config, save_name)
    models.append(norm_base)

# init data
human_ref = models[0].r
human_tun1, human_tun2 = models[0].t_mean[[1,2]]
human_cat1 = human_tun1 + human_ref
human_cat2 = human_tun2 + human_ref
monkey_ref = models[1].r
monkey_tun1, monkey_tun2 = models[1].t_mean[[1,2]]
monkey_cat1 = monkey_ref + monkey_tun1
monkey_cat2 = monkey_ref + monkey_tun2
both_ref = (human_ref + monkey_ref) /2
both_cat1 = (human_cat1 + monkey_cat1) /2
both_cat2 = (human_cat2 + monkey_cat2) /2
both_tun1 = (human_tun1 + monkey_tun1) /2
both_tun2 = (human_tun2 + monkey_tun2) /2

# print distances
print("reference distances")
points = [human_ref, monkey_ref, both_ref]
for i, point1 in enumerate(points):
    for j, point2 in enumerate(points):
        if j<i: continue
        print("distance {} to {}:".format(i,j), np.linalg.norm(point1-point2))
print()

# print distances
print("reference and cat1 distances")
points = [human_ref, human_cat1, monkey_ref, monkey_cat1, both_ref, both_cat1]
for i, point1 in enumerate(points):
    for j, point2 in enumerate(points):
        if j < i: continue
        print("distance {} to {}:".format(i,j), np.linalg.norm(point1-point2))
print()

# print angle between tuning vectors
# Note: returns angle in radians
def theta(v, w): return np.arccos(v.dot(w)/(np.linalg.norm(v)*np.linalg.norm(w))) /np.pi * 180

print("angles between tuning vectors for category 1")
vectors = [human_tun1, monkey_tun1, both_tun1]
for i, vector1 in enumerate(vectors):
    for j, vector2 in enumerate(vectors):
        if j <= i: continue
        print("angle between {} and {}".format(i,j), theta(vector1, vector2))
print()
print("angles between tuning vectors for category 2")
vectors = [human_tun2, monkey_tun2, both_tun2]
for i, vector1 in enumerate(vectors):
    for j, vector2 in enumerate(vectors):
        if j <= i: continue
        print("angle between {} and {}".format(i,j), theta(vector1, vector2))