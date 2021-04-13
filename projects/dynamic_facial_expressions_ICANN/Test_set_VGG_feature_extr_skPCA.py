import numpy as np
import pickle
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.load_data_EB import load_data
from utils.Load_model_1 import _load_model


def load_EB_data(config):
    #Load data
    # todo modify the train=True but for now it#s the easiest to modify the config file with the csv to load
    data = load_data(config, train=True) #Program will load val/test if you set #Train = False
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
    print()
    return data


def extract_v4_features(data, config, save=True):
    #Use pre-trained VGG to extract mid-level features
    v4 = _load_model(config, input_shape=(224, 224, 3))
    # #print(v4.summary())
    print("[INIT] -- Model loaded --")
    print("[INIT] Model:", config['model'])
    print("[INIT] V4 layer:", config['v4_layer'])

    shape_v4 = np.shape(v4.layers[-1].output)
    n_features = shape_v4[1] * shape_v4[2] * shape_v4[3]
    print("[INIT] n_features:", n_features)

    print("[INIT] Load PCA")
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    n_category = config['n_category']  # Number of categories/conditions in the test set

    print("[V4] -- V4 Prediction --")
    x = tf.keras.applications.vgg19.preprocess_input(np.copy(data))
    # print(preds)
    preds = v4.predict(x)
    preds = np.reshape(preds, (data.shape[0], -1))
    print("[V4] Shape prediction", preds.shape)
    print("[V4] min max preds:", np.amin(preds), np.amax(preds))
    #print(features[:,0:25])

    #Scale the data before applying pca
    #features = sc.transform(features)

    print("[Feature] -- feature reduction --")
    print("[Feature] load max feature")
    max_feature = np.load("max_feature.npy")
    features = preds/max_feature
    print("[Feature] min max Feature:", np.amin(features), np.amax(features))

    pca_red_features_test = pca.transform(features)
    print("[PCA] features reduced to:", np.shape(pca_red_features_test))
    print("[PCA] min max feature", np.amin(pca_red_features_test), np.amax(pca_red_features_test))

    if save:
        #Save as a pickle file
        #pickle.dump(TSDATARR, open('Test_seq_features.pkl', 'wb'))
        with open('Test_seq_features.pkl', 'wb') as f:
            pickle.dump([pca_red_features_test, n_category], f)

