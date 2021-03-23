import numpy as np
import os
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
# import fiftyone  # has a nice API but seems difficult to get the segmentation


"""
helper function to load annotation following the COCO format: https://github.com/cocodataset/cocoapi

the protocol was made following this: https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
"""


def getClassName(classID, cats):
    """
    heleper function to retrieve class name from the class ID

    :param classID:
    :param cats:
    :return:
    """
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


def load_coco_semantic_annotations(config, verbose=False):
    """
    build semantic labels from labeled masks following the "coco" dataset format

    :return:
    """

    path = config["semantic_img_path"]
    json_path = os.path.join(path, config["coco_annot_json"])
    # Initialize the COCO api for instance annotations
    coco = COCO(json_path)

    # Load the categories in a variable
    cat_ids = coco.getCatIds()
    categories = coco.loadCats(cat_ids)

    # get all image ids
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)

    # retrieves parameters
    num_images = len(images)
    num_labels = len(categories)
    image_size = tuple(config["image_size"])

    # declare variables
    x = np.zeros((num_images, image_size[0], image_size[1], 3))
    labels = np.zeros((num_images, image_size[0], image_size[1], num_labels))

    # print information
    if verbose:
        print("categories")
        print(categories)
        print("image ids")
        print(image_ids)
        print("shape x", np.shape(x))
        print("shape labels", np.shape(labels))

    # build semantic labels
    for i, image in enumerate(images[:1]):
        # load image
        im = cv2.imread(os.path.join(path, "data", image["file_name"]))
        im = cv2.resize(im, image_size)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x[i, :, :, :] = im_rgb

        # get all annotations of the image
        annIds = coco.getAnnIds(imgIds=image['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(annIds)

        for a in range(len(anns)):
            # # get class name
            # class_name = getClassName(anns[a]['category_id'], categories)

            # create mask
            mask = np.zeros((image['height'], image['width']))

            # fill mask
            mask = np.maximum(coco.annToMask(anns[a]), mask)

            # get category index
            cat_id = anns[a]['category_id'] - 1  # ids start at 1

            # resize mask
            mask = cv2.resize(mask, image_size)

            # add mask
            labels[i, :, :, cat_id] += mask  # += since some annotations could appears multiple times per images

            # # show mask
            # plt.imshow(mask)
            # plt.show()

    # since we do +=, simply set all values >1 back to 1
    labels[labels > 1] = 1

    return [x, labels]


def load_coco_categories(config, verbose=False):
    # load coco to get the categories' name
    path = config["semantic_img_path"]
    json_path = os.path.join(path, config["coco_annot_json"])
    # Initialize the COCO api for instance annotations
    coco = COCO(json_path)

    # Load the categories in a variable
    cat_ids = coco.getCatIds()
    categories = coco.loadCats(cat_ids)

    if verbose:
        for c in range(len(categories)):
            # print category name
            print("categories", categories[c])

    return categories


def get_coco_cat_ids(config, categories, to_numpy=False):
    # load coco to get the categories' name
    path = config["semantic_img_path"]
    json_path = os.path.join(path, config["coco_annot_json"])
    # Initialize the COCO api for instance annotations
    coco = COCO(json_path)

    cat_ids = np.array(coco.getCatIds(catNms=categories))
    if to_numpy:
        cat_ids = cat_ids - 1

    return cat_ids


if __name__ == "__main__":
    """
    Test load_coco_semantic_annotation function
    
    run: python -m utils.load_coco_semantic_annotations
    """
    # todo create a COCO class ?

    from utils.load_config import load_config

    # load config
    config_path = 'CNN_t02_find_semantic_units_m0001.json'
    config = load_config(config_path, path='configs/CNN')

    # test function
    data = load_coco_semantic_annotations(config)
    x = data[0]
    labels = data[1]

    # Load the categories in a variable
    categories = load_coco_categories(config)
    # print all masks per images
    for i, label in enumerate(labels[:1]):
        for c in range(len(categories)):
            # print category name
            print("categories", categories[c])

            # display mask
            plt.imshow(label[:, :, c])
            plt.show()
