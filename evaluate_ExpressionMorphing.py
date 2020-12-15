from utils.load_config import load_config
from utils.load_data import load_data

import cv2
import os

config = load_config("norm_base_expressionMorphing_t0001.json")

#path = "/app/Data/Dataset/ExpressionMorphing/images/HumanAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0.0000.png"
#im_test = cv2.imread("/app/Data/Dataset/ExpressionMorphing/ExpressionMorphing/images/HumanAvatar_Anger_0.0_Fear_1.0_Monkey_1.0_Human_0.0.0000.png")
#print(os.path.exists(path))

test = load_data(config)