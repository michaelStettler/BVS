import os
import numpy as np

from utils.load_config import load_config
from utils.load_data import load_data
from models.ExampleBase import ExampleBase

"""
Reproduce the results from the ICANN paper but with the updated VGG pipeline

run: python -m projects.dynamic_facial_expressions_ICANN.03_reproduce_ICANN_ExampleBase
"""

# load config
config = load_config("EB_reproduce_ICANN_cat.json", path="configs/example_base")

# load model
model = ExampleBase(config, input_shape=tuple(config['input_shape']), load_EB_model=True)

# load data
train_data = load_data(config)
# segment sequence based on config
seg_data = []
if config.get('concat_seg_start') is not None:
    for start in config['concat_seg_start']:
        seg_data.append(train_data[0][start:start + config['batch_size']])
seg_data = np.array(seg_data)
seg_data = np.reshape(seg_data, (-1, seg_data.shape[2], seg_data.shape[3], seg_data.shape[4]))
print("shape segmentated data", np.shape(seg_data))
train_data[0] = seg_data

# fit model
model.fit(train_data,
          fit_normalize=False,
          fit_dim_red=False,
          fit_snapshots=False)
model.save()

# --------------------------------------------------------------------------------------------------------------------
# plot training

# plot training snapshots
model.plot_snapshots(title="01_train")

# plot neural field
model.plot_neural_field(title="02_train")
