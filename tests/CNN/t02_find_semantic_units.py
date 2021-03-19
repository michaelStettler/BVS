import os
import numpy as np
# import fiftyone  # has a nice API but seems difficult to get the segmentation
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from utils.load_config import load_config
from utils.load_data import load_data
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

# todo make everxthing within this layout
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


path = '/Users/michaelstettler/Desktop/affectnet_face_semantic'
json_name = 'AffectNet_semantic_COCO.json'
json_path = os.path.join(path, json_name)

# Initialize the COCO api for instance annotations
coco = COCO(json_path)

# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats)

# todo probably to put in a utils
def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"
print('The class name is', getClassName(2, cats))


# Load and display instance annotations
plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
