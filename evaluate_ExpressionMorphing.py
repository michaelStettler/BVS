from utils.load_config import load_config
from utils.load_data import load_data

import cv2
import os

config = load_config("norm_base_expressionMorphing_t0001.json")

data_train = load_data(config)
data_test = load_data(config, train=False)