import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

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
    n_classes = 3
    dataX = np.random.rand(n_data, 224, 224, 3)
    dataY = np.random.randint(n_classes, size=n_data)
    dataY = np.eye(n_classes)[dataY.reshape(-1)]   # transform to one hot encoding

else:
    raise ValueError("training data: {} does not exists! Please change config file or add the training data"
                     .format(config['train_data']))

print("[Data] -- Data loaded --")
print("[Data] shape dataX", np.shape(dataX))
print("[Data] shape dataY", np.shape(dataY))

# load model
if config['model'] == 'VGG16':
    model = tf.keras.applications.VGG16(include_top=False,
                                        weights="imagenet")
    v4 = Model(inputs=model.input,
               outputs=model.get_layer(config['v4_layer']).output)
else:
    raise ValueError("model: {} does not exists! Please change config file or add the model"
                     .format(config['model']))

print("[Model] -- Model loaded --")
print("[Model] V4 layer:", config['v4_layer'])
print(v4.summary())

# predict images
preds = v4.predict(dataX)
print("shape preds", np.shape(preds))
