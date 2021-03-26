import os
import numpy as np
import tensorflow as tf
from utils.load_config import load_config
from utils.load_extraction_model import load_extraction_model
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

# define configuration
config_path = 'CNN_t03_feature_map_positions_m0001.json'

# load config
config = load_config(config_path, path='configs/CNN')

# choose feature map index
eyebrow_ft_idx = 148   # the index come from the find_semantic_units

# load and define model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(model.output)[1:3])
print("size_ft", size_ft)

# ----------------------------------------------------------------------------------------------------------------------
# test 1 - control weighted average
# define feature map controls
num_entry = 3
num_ft = 1
preds1 = np.zeros((num_entry,) + size_ft + (num_ft,))
print("shape preds1", np.shape(preds1))
# create test with 1 at each corner
preds1[1, 0, 0, 0] = 1
preds1[1, 0, -1, 0] = 1
preds1[1, -1, 0, 0] = 1
preds1[1, -1, -1, 0] = 1

# compute position vectors
preds1_pos = calculate_position(preds1, mode="weighted average", return_mode="array")
print("shape preds1_pos", np.shape(preds1_pos))
# plot positions
plot_cnn_output(preds1_pos, os.path.join("models/saved", config["config_name"]),
                config['v4_layer'] + "_test1.gif",
                image=preds1,
                video=True,
                verbose=False)
print("[TEST 1] Finished plotting dynamic eyebrow selection")

# # predict
# preds = model.predict(data)
