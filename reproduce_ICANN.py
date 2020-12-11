import numpy as np
import os

from utils.load_data import load_data
from utils.load_config import load_config
from models.NormBase import NormBase


config = load_config("norm_base_expressivityLevels_t0003.json")
save_name = "simplified"

data_train = load_data(config)

# fit and save model
norm_base = NormBase(config, input_shape=(224,224,3))
norm_base.fit(data_train)
norm_base.save_model(config, save_name)

#load model
norm_base = NormBase(config, input_shape=(224,224,3), save_name=save_name)
data_test = load_data(config, train=False, sort_by=['image'])

# evaluate model
accuracy1, it_resp1, labels1 = norm_base.evaluate_accuracy(data_train)
accuracy2, it_resp2, labels2 = norm_base.evaluate_accuracy(data_test)

print("accuracy1", accuracy1)
print("it_resp1.shape", it_resp1.shape)
print("labels1.shape", labels1.shape)
print("accuracy2", accuracy2)
print("it_resp2.shape", it_resp2.shape)
print("labels2.shape", labels2.shape)

accuracy = (0.25*accuracy1)+(0.75*accuracy2)
it_resp = np.concatenate([it_resp1, it_resp2], axis=0)
labels = np.concatenate([labels1,labels2], axis=0)

print("accuracy", accuracy)
print("it_resp.shape", it_resp.shape)
print("labels.shape", labels.shape)

# save results to be used by reproduce_ICANN_plot.py
save_folder = os.path.join("models/saved", config['save_name'], save_name)
np.save(os.path.join(save_folder, "accuracy"), accuracy)
np.save(os.path.join(save_folder, "it_resp"), it_resp)
np.save(os.path.join(save_folder, "labels"), labels)