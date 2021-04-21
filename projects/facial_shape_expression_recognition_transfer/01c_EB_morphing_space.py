import numpy as np
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.ExampleBase import ExampleBase
from datasets_utils.morphing_space import transform_morph_space_list2space
from plots_utils.plot_morphing_space import plot_morphing_space

"""
test the naive PCA selection on the original morphing space dataset from the eLife paper

run: python -m projects.facial_shape_expression_recognition_transfer.01c_EB_morphing_space
"""

train_model = False
predict_it_resp = False

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
    train_expr_neuron, strain_naps, train_nn_field = model.fit(data,
                                             fit_normalize=True,
                                             fit_dim_red=True,
                                             fit_snapshots=True,
                                             get_snapshots=True,
                                             get_nn_field=True)
    model.save()

    # plot training snapshots
    model.plot_snapshots(strain_naps, title="01_train")

    # plot neural field kernels
    model.plot_nn_kernels(title="02_train")

    # plot neural field
    model.plot_neural_field(train_nn_field, title="03_train")

    # plot expression neurons
    model.plot_expression_neurons(train_expr_neuron, title="04_train")

# --------------------------------------------------------------------------------------------------------------------
# predict model
if predict_it_resp:
    print("[Test] -- Predict model] --")
    data = load_data(config, train=False)
    expr_neuron, snaps, nn_field = model.predict(data, get_snapshots=True, get_nn_field=True)
    print("[Test] model predicted")
    print("[Test] shape snaps", np.shape(snaps))
    print("[Test] shape nn_field", np.shape(nn_field))
    print("[Test] shape expr_neuron", np.shape(expr_neuron))

    np.save(os.path.join("models/saved", config['config_name'], "ExampleBase", "snaps_neuron"), snaps)
    np.save(os.path.join("models/saved", config['config_name'], "ExampleBase", "expr_neuron"), expr_neuron)
    np.save(os.path.join("models/saved", config['config_name'], "ExampleBase", "nn_field"), nn_field)
    print("[Test] snapshots and expr_neuron saved!")
else:
    snaps = np.load(os.path.join("models/saved", config['config_name'], "ExampleBase", "snaps_neuron.npy"))
    expr_neuron = np.load(os.path.join("models/saved", config['config_name'], "ExampleBase", "expr_neuron.npy"))
    nn_field = np.load(os.path.join("models/saved", config['config_name'], "ExampleBase", "nn_field.npy"))
    print("[Test] response neuron loaded!")
print("[Test] shape expr_neuron", np.shape(expr_neuron))
print("[Test] shape snaps_neuron", np.shape(snaps))
print()


# # plot training snapshots
# model.plot_snapshots(snaps, title="01_test")
#
# # # plot neural field kernels
# # model.plot_nn_kernels(title="02_test")
#
# # plot neural field
# model.plot_neural_field(nn_field, title="03_test")
#
# # plot expression neurons
# model.plot_expression_neurons(expr_neuron, title="04_test")

# plot morphing space
expr_neuron_morph_space = transform_morph_space_list2space(expr_neuron)
print("[Plot] shape morphing space", np.shape(expr_neuron_morph_space))
# it_morph_space = np.amax(it_morph_space, axis=2)  # compute max over all sequences
it_morph_space = np.sum(expr_neuron_morph_space, axis=2)  # compute max over all sequences
print("[Plot] shape it_morph_space", np.shape(it_morph_space))
plot_morphing_space(it_morph_space,
                    save_folder=os.path.join(os.path.join("models/saved", config['config_name'])))
