do_plot = 1 #Added for plotting
do_nice = 0 #Added for plotting
import matplotlib.pyplot as plt #Added for plotting
import numpy as np
import pickle
import json
import os
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.load_data_EB import load_data
from utils.Load_model_1 import _load_model

# matplotlib.use('Agg')  # Added for plotting
plt.style.use('seaborn-paper')  # Added for plotting

pca = PCA(n_components=25)
#or
# pca = PCA(0.95)

config_path = 'configs/example_base_config'
# config_name = 'example_base_reproduce_ICANN_cat.json'
config_name = 'example_base_reproduce_ICANN_expressivity.json'
#config_name = 'example_base_monkey_test_reduced_dataset_.json'

config_file_path = os.path.join(config_path, config_name)
print("config_file_path", config_file_path)

#Load example_base_config file
with open(config_file_path) as json_file:
    config = json.load(json_file)

#Load data
data = load_data(config, train=True)
print("[Data] -- Data loaded --")
# Use relevant part of data for feature extraction
data = data[0]
print("[Data] Shape data", np.shape(data))

# segment sequence based on config
seg_data = []
if config.get('concat_seg_start') is not None:
    for start in config['concat_seg_start']:
        seg_data.append(data[start:start + config['batch_size']])
seg_x = np.array(seg_data)
data = np.reshape(seg_x, (-1, seg_x.shape[2], seg_x.shape[3], seg_x.shape[4]))
print("[Data] sequences segmented, new data shape:", np.shape(data))

#Use pre-trained VGG to extract mid-level features
v4 = _load_model(config, input_shape=(224, 224, 3))
# #print(v4.summary())
print("[INIT] -- Model loaded --")
print("[INIT] Model:", config['model'])
print("[INIT] V4 layer:", config['v4_layer'])
shape_v4 = np.shape(v4.layers[-1].output)
n_features = shape_v4[1] * shape_v4[2] * shape_v4[3]
print("[INIT] n_features:", n_features)

#Extract features using VGG-19
x = tf.keras.applications.vgg19.preprocess_input(np.copy(data))
# print(preds)
preds = v4.predict(x)
preds = np.reshape(preds, (data.shape[0], -1))
print("Shape of features after extraction and reshaping", preds.shape)
print("V4 preds, min max:", np.amin(preds), np.amax(preds))
#print(features[:,0:25])

# #Scale the data before applying pca
# sc = StandardScaler()
# sc.fit(features)
#
# #Save scaler to use on test data
# pickle.dump(sc, open('scaler.pkl','wb'))
#
# #Scale training data
# features = sc.transform(features)
# print("Scaled features, min max:", np.amin(features), np.amax(features))
# # print(features)
# print('max elems')
# print(np.max(features, axis=0))

# normalize data
# features = features - np.mean(features)
max_features = np.amax(preds)
features = preds / max_features
print("normalized features, min max:", np.amin(features), np.amax(features))
print("save max features")
np.save("max_feature", max_features)

#Fit pca
pca.fit(features)

#Store pca in a pickle file to use on test data
pickle.dump(pca, open('pca.pkl', 'wb'))

#Transform training data before reshaping
pca_red_features = pca.transform(features)
print("[PCA] min max feature", np.amin(pca_red_features), np.amax(pca_red_features))
print("PCA Transformed features")
# print(pca_red_features)
print("explained variance", pca.explained_variance_ratio_)
print("explained variance", np.sum(pca.explained_variance_ratio_))
print("PCA SINGULAR VALUES__")
print(pca.singular_values_)

singular_val = pca.singular_values_
print("Shape of singular value array")
print(singular_val.shape)
singular_val = np.expand_dims(singular_val, axis=1)

print("Shape of transformed data")
print(features.shape)

#pca_red_features = features[:,0:25]
print(pca_red_features.shape)
print(pca_red_features)

#pca_red_features = (pca_red_features/1000)
#print(pca_red_features)

#Save as a pickle file
pickle.dump(pca_red_features, open('Train_seq_features.pkl', 'wb'))

if do_plot > 0: #Currently handling only 1 do_nice case (do_nice = 0 case from matlab)
    plt.subplot(2, 1, 1)
    plt.plot(singular_val[:100, 0])
    plt.title('Singular values')
        
    plt.subplot(2, 1, 2)
    plt.imshow(np.transpose(pca_red_features))
    plt.title('Whitened reduced features')
    print('PCA figure - saving')
    plt.savefig('Singular values.png')
