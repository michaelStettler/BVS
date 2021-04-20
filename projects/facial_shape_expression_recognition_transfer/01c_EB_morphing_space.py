import numpy as np
import os

from datasets_utils.tranform_morph_space_list2space import transform_morph_space_list2space
from utils.load_config import load_config
from utils.load_data import load_data
from models.ExampleBase import ExampleBase

"""
test the naive PCA selection on the original morphing space dataset from the eLife paper

run: python -m projects.facial_shape_expression_recognition_transfer.01c_EB_morphing_space
"""

train_model = False
predict_it_resp = True

# load config
config_name = 'EB_morphing_space_m0001.json'
config = load_config(config_name, path='configs/example_base')

# --------------------------------------------------------------------------------------------------------------------
# declare model
load_model = False
if not train_model:
    load_model = True
model = ExampleBase(config, input_shape=tuple(config['input_shape']), load_EB_model=load_model)

# train model
if train_model:
    data = load_data(config, train=True)
    print("[FIT] -- Train model --")
    expr_neuron, snaps = model.fit(data,
                                   fit_normalize=True,
                                   fit_dim_red=True,
                                   fit_snapshots=True,
                                   get_snapshots=True)
    model.save()

    # plot training snapshots
    model.plot_snapshots(snaps, title="01_train")

    # plot neural field
    model.plot_neural_field(title="02_train")

# --------------------------------------------------------------------------------------------------------------------
# predict model
if predict_it_resp:
    print("[Test] -- Predict model] --")
    data = load_data(config, train=False)
    expr_neuron, snaps = model.predict(data, get_snapshots=True)
    print("[Test] model predicted")
    np.save(os.path.join("models/saved", config['config_name'], "ExampleBase", "snaps_neuron"), snaps)
    np.save(os.path.join("models/saved", config['config_name'], "ExampleBase", "expr_neuron"), expr_neuron)
    print("[Test] snapshots and expr_neuron saved!")
else:
    snaps = np.load(os.path.join("models/saved", config['config_name'], "ExampleBase", "snaps_neuron.npy"))
    expr_neuron = np.load(os.path.join("models/saved", config['config_name'], "ExampleBase", "expr_neuron.npy"))
    print("[Test] expr_neuron loaded!")
print("[Test] shape expr_neuron", np.shape(expr_neuron))
print("[Test] shape snaps_neuron", np.shape(snaps))
print()


# plot training snapshots
model.plot_snapshots(snaps, title="01_test")

# plot neural field
model.plot_neural_field(title="02_test")
