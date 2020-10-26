import json
import os
import numpy as np
import tensorflow as tf

from models.NormBase import NormBase

congig_path = 'model_config'
config_name = 'norm_base_test.json'
config_file_path = os.path.join(congig_path, config_name)
print("config_file_path", config_file_path)

# load model_config file
with open(config_file_path) as json_file:
    config = json.load(json_file)

# load data
if config['train_data'] == 'test':
    print("generate random training data")
    np.random.seed(0)
    n_data = 36
    n_classes = 3
    dataX = np.random.rand(n_data, 224, 224, 3)
    dataY = np.random.randint(n_classes, size=n_data)
    # dataY = np.eye(n_classes)[dataY.reshape(-1)]   # transform to one hot encoding

else:
    raise ValueError("training data: {} does not exists! Please change model_config file or add the training data"
                     .format(config['train_data']))

print("[Data] -- Data loaded --")
print("[Data] shape dataX", np.shape(dataX))
print("[Data] shape dataY", np.shape(dataY))

# create model
norm_base = NormBase(config, input_shape=(224, 224, 3))
# train model
norm_base.fit(x=dataX, y=dataY, batch_size=int(config['batch_size']))
