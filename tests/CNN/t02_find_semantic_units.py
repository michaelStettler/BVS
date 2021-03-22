import os
import numpy as np
# import fiftyone  # has a nice API but seems difficult to get the segmentation
import skimage.io as io
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from utils.load_config import load_config
from utils.load_coco_semantic_annotations import load_coco_semantic_annotations
from utils.load_extraction_model import load_extraction_model
from utils.find_semantic_units import find_semantic_units

"""
test to try the find_semantic_function that implement the paper:
todo put paper name

the COCO protocol is following this: https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047

run: python -m tests.CNN.t02_find_semantic_units
"""
np.random.seed(0)
config_path = 'CNN_t02_find_semantic_units_m0001.json'
save = True

np.set_printoptions(precision=3, suppress=True, linewidth=150)

config = load_config(config_path, path='configs/CNN')

# todo make everything within this layout
# # load model
# model = load_extraction_model(config)
# # print(model.summary())
#
# # load data
# data = load_data(config)
# print("[Loading] shape x", np.shape(data[0]))
# print("[Loading] shape label", np.shape(data[1]))
# print("[loading] finish loading data")
# print()
#
# # compute face units
# find_semantic_units(model, data[0], data[1])


data = load_coco_semantic_annotations(config)
x = data[0]
labels = data[1]

# print all masks per images
for i, label in enumerate(labels[:1]):
    for c in range(len(categories)):
        # print category name
        print("categories", categories[c])

        # display mask
        plt.imshow(label[:, :, c])
        plt.show()


