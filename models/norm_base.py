import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

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

# load model
if config['model'] == 'VGG16':
    model = tf.keras.applications.VGG16(include_top=False,
                                        weights="imagenet")
    v4 = Model(inputs=model.input,
               outputs=model.get_layer(config['v4_layer']).output)
else:
    raise ValueError("model: {} does not exists! Please change model_config file or add the model"
                     .format(config['model']))

print("[Model] -- Model loaded --")
print("[Model] Architecture:", config['model'])
print("[Model] V4 layer:", config['v4_layer'])
# print(v4.summary())


print()
print("----------- cumulative batch computing ------------")
print("[NormBase] Compute ref vector m")
num_iter = int(n_data/config['batch_size'])
# predict first image to get the size
shape_preds = np.shape(v4.predict(np.expand_dims(dataX[0], axis=0)))
n_features = np.prod(shape_preds)
print("[NormBase] n_features", n_features)
m = np.zeros(n_features)
m_cumul = 0
for b in range(num_iter):
    # -----------------------------------------------------------------------------
    # get batch entry
    start = b * config['batch_size']
    end = start + config['batch_size']
    batch_data = dataX[start:end, :]
    batch_label = dataY[start:end]

    # keep only reference data
    # index 0 represent the refeence category #todo select ref category?
    ref_data = batch_data[batch_label == 0]

    # predict ref batch
    pred = v4.predict(ref_data)
    pred = np.reshape(pred, (np.shape(ref_data)[0], -1))

    # -----------------------------------------------------------------------------
    # update ref_vector m
    n_ref = np.shape(ref_data)[0]
    m = (m_cumul*m + n_ref*np.mean(pred, axis=0))/(m_cumul + n_ref)
    m_cumul += n_ref
print("[NormBase] Ref. vector computed! Shape ref. vector m:", np.shape(m))

# compute direction tunning for each class based on ref vector m
print("[NormBase] Compute direction tuning n")
n_mean = np.zeros((n_classes, n_features))
n = np.zeros((n_classes, n_features))
n_cumul = np.zeros(n_classes)
m_batch = np.repeat(np.expand_dims(m, axis=1), config['batch_size'], axis=1).T
for b in range(num_iter):
    # -----------------------------------------------------------------------------
    # get batch entry
    start = b * config['batch_size']
    end = start + config['batch_size']
    batch_data = dataX[start:end, :]
    batch_label = dataY[start:end]

    # predict ref batch
    pred = v4.predict(batch_data)
    pred = np.reshape(pred, (config['batch_size'], -1))
    # -----------------------------------------------------------------------------
    # update direction tuning vector n

    # compute difference u - m with ref vector m
    batch_diff = pred - m_batch

    # compute direction tuning for each category
    for i in range(n_classes):
        # get data for each category
        cat_diff = batch_diff[batch_label == i]
        # get num_of data
        n_cat_diff = np.shape(cat_diff)[0]
        # update cumulative mean for each category
        n_mean[i] = (n_cumul[i]*n_mean[i] + n_cat_diff*np.mean(cat_diff, axis=0))/(n_cumul[i] + n_cat_diff)
        # update cumulative counts
        n_cumul[i] += n_cat_diff

        n[i] = n_mean[i] / np.linalg.norm(n_mean[i])

print("[NormBase] Ref. vector computed! Shape ref. vector n:", np.shape(n))
