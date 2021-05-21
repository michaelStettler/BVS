import os
import numpy as np
import tensorflow as tf

from models.RBF import RBF
from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from plots_utils.plot_cnn_output import plot_cnn_output

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the fit a face template over different feature maps instead of simply summing them up

run: python -m tests.CNN.t03b_feature_map_face_templates
"""

# define configuration
config_path = 'CNN_t03b_feature_map_face_templates_m0001.json'

# declare parameters
best_eyebrow_IoU_ft = [209, 148, 59, 208]
best_lips_IoU_ft = [77, 79, 120, 104, 141, 0, 34, 125, 15, 89, 49, 237, 174, 39, 210, 112, 111, 201, 149, 165, 80,
                         42, 128, 74, 131, 193, 133, 44, 154, 101, 173, 6, 148, 61, 27, 249, 209, 19, 247, 90, 1, 255,
                         182, 251, 186, 248]

# load config
config = load_config(config_path, path='configs/CNN')

# create directory if non existant
save_path = os.path.join("models/saved", config["config_name"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

# load and define model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(model.output)[1:3])
print("[LOAD] size_ft", size_ft)
print("[LOAD] Model loaded")
print()

# load data
data = load_data(config)

# predict
preds = model.predict(data[0], verbose=1)
print("[PREDS] shape prediction", np.shape(preds))

# normalize predictions
preds /= np.amax(preds)

# --------------------------------------------------------------------------------------------------------------------
# analyse feature maps

# -------------------------------------------------------------------------------------------------------------------
# eyebrow

# keep only highest eyebrow IoU semantic score
eyebrow_preds = preds[..., best_eyebrow_IoU_ft]
print("shape semantic feature selection", np.shape(preds))

# compute average eyebrow feature maps
ft_average = np.mean(eyebrow_preds, axis=-1)
ft_average = np.expand_dims(ft_average, axis=3)
print("shape ft_average", np.shape(ft_average))

# concacenate feature maps for plotting
eyebrow_preds = np.concatenate((eyebrow_preds, ft_average), axis=-1)
print("shape concacenate eyebrow_preds", np.shape(eyebrow_preds))

# build eyebrow template
eyebrow_mask = np.array([[7, 12], [8, 21]])
eyebrow_template = eyebrow_preds[0, eyebrow_mask[0, 0]:eyebrow_mask[0, 1], eyebrow_mask[1, 0]:eyebrow_mask[1, 1]]  # neutral
# eyebrow_template = preds[60, eyebrow_mask[0, 0]:eyebrow_mask[0, 1], eyebrow_mask[1, 0]:eyebrow_mask[1, 1]]  # max C2
eyebrow_template = np.expand_dims(eyebrow_template, axis=0)

# test mask positions
# eyebrow_preds[:, eyebrow_mask[0, 0]:eyebrow_mask[0, 1], eyebrow_mask[1, 0]:eyebrow_mask[1, 1]] = 1

# apply rbf kernel
eyebrow_rbf = RBF(config)
eyebrow_rbf.fit2d(eyebrow_template)
eyebrow_ft = eyebrow_rbf.predict2d(eyebrow_preds)
print("shape eyebrow_ft", np.shape(eyebrow_ft))

# concacenate feature maps for plotting
eyebrow_preds = np.concatenate((eyebrow_preds, eyebrow_ft), axis=-1)
print("shape concacenate eyebrow_preds", np.shape(eyebrow_preds))
print()

# -------------------------------------------------------------------------------------------------------------------
# lips

# keep only highest IoU semantic score
lips_preds = preds[..., best_lips_IoU_ft]

# compute average lips feature maps
ft_average = np.mean(lips_preds, axis=-1)
ft_average = np.expand_dims(ft_average, axis=3)
print("shape ft_average", np.shape(ft_average))

# concacenate feature maps for plotting
lips_preds = np.concatenate((lips_preds, ft_average), axis=-1)
print("shape concacenate lips_preds", np.shape(lips_preds))

# build lips template
lips_mask = np.array([[15, 19], [10, 19]])
lips_template = lips_preds[0, lips_mask[0, 0]:lips_mask[0, 1], lips_mask[1, 0]:lips_mask[1, 1]]  # neutral
lips_template = np.expand_dims(lips_template, axis=0)

# test mask positions
# lips_preds[:, lips_mask[0, 0]:lips_mask[0, 1], lips_mask[1, 0]:lips_mask[1, 1]] = 1

# apply rbf kernel
lips_rbf = RBF(config)
lips_rbf.fit2d(lips_template)
lips_ft = lips_rbf.predict2d(lips_preds)
print("shape lips_ft", np.shape(lips_preds))

# concacenate feature maps for plotting
lips_preds = np.concatenate((lips_preds, lips_ft), axis=-1)
print("shape concacenate preds", np.shape(lips_preds))


# --------------------------------------------------------------------------------------------------------------------
# plots

# eyebrow feature maps
plot_cnn_output(eyebrow_preds, os.path.join("models/saved", config["config_name"]),
                "eyebrow_feature_maps_output.gif", verbose=True, video=True)

# plot dynamic
eyebrow_preds_ref = eyebrow_preds[0]
eyebrow_preds_dyn = eyebrow_preds - np.repeat(np.expand_dims(eyebrow_preds_ref, axis=0), len(eyebrow_preds), axis=0)
eyebrow_preds_dyn[eyebrow_preds_dyn < 0] = 0

plot_cnn_output(eyebrow_preds_dyn, os.path.join("models/saved", config["config_name"]),
                "eyebrow_dyn_feature_maps_output.gif", verbose=True, video=True)

# lips feature maps
plot_cnn_output(lips_preds, os.path.join("models/saved", config["config_name"]),
                "lips_feature_maps_output.gif", verbose=True, video=True)

# plot dynamic
lips_preds_ref = lips_preds[0]
lips_preds_dyn = lips_preds - np.repeat(np.expand_dims(lips_preds_ref, axis=0), len(lips_preds), axis=0)
lips_preds_dyn[lips_preds_dyn < 0] = 0

plot_cnn_output(lips_preds_dyn, os.path.join("models/saved", config["config_name"]),
                "lips_dyn_feature_maps_output.gif", verbose=True, video=True)
