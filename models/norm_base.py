import json
import os
import numpy as np
import tensorflow as tf

congig_path = 'config'
config_name = 'norm_base_test.json'
config_file_path = os.path.join(congig_path, config_name)
print("config_file_path", config_file_path)

# load config file
with open(config_file_path) as json_file:
    config = json.load(json_file)

# load data
if config['train_data'] == 'test':
    print("generate random training data")
    np.random.seed(0)
    n_data = 36
    n_cat = 3
    dataX = np.random.rand(n_data, 34, 34, 3)
    dataY = np.random.randint(n_cat, size=n_data)
    dataY = np.eye(n_cat)[dataY.reshape(-1)]   # transform to one hot encoding

else:
    raise ValueError("training data: {} does not exists! Please change config file or add the training data"
                     .format(config['train_data']))

print("[Data] -- data loaded --")
print("[Data] shape dataX", np.shape(dataX))
print("[Data] shape dataY", np.shape(dataY))
