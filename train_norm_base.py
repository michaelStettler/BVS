import json
import os
import numpy as np
import tensorflow as tf

from models.NormBase import NormBase

congig_path = 'configs'
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
    n_classes = config['n_category']
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
# norm_base.print_v4_summary()

# train model
m, n = norm_base.fit(x=dataX, y=dataY, batch_size=config['batch_size'])

print("shape m", np.shape(m))
print("shape n", np.shape(n))


# save_folder = os.path.join("models/saved", config['save_name'])
# if not os.path.exists(save_folder):
#     os.mkdir(save_folder)
# np.save(os.path.join(save_folder, "ref_vector"), m)
# np.save(os.path.join(save_folder, "tuning_vector"), n)


