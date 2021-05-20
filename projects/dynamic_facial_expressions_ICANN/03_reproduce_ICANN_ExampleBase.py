import os
import numpy as np

from utils.load_config import load_config
from utils.load_data import load_data
from models.ExampleBase import ExampleBase
from datasets_utils.expressivity_level import segment_sequence
from plots_utils.plot_expressivity_space import plot_expressivity_level

"""
Reproduce the results from the ICANN paper but with the updated VGG pipeline

run: python -m projects.dynamic_facial_expressions_ICANN.03_reproduce_ICANN_ExampleBase
"""

# load config
# config = load_config("EB_reproduce_ICANN_cat.json", path="configs/example_base")
config = load_config("EB_reproduce_ICANN_expressivity.json", path="configs/example_base")

# load model
model = ExampleBase(config, input_shape=tuple(config['input_shape']), load_EB_model=False)

# --------------------------------------------------------------------------------------------------------------------
# train model

# load data
train_data = load_data(config)
train_data[0] = segment_sequence(train_data[0], config['train_seg_start_idx'], config['seq_length'])
train_data[1] = segment_sequence(train_data[1], config['train_seg_start_idx'], config['seq_length'])
print("[LOAD] Shape train_data segmented", np.shape(train_data[0]))
print("[LOAD] Shape train_label segmented", np.shape(train_data[1]))

# fit model
expr_resp, snaps, nn_field = model.fit(train_data,
                                       fit_normalize=True,
                                       fit_dim_red=True,
                                       fit_snapshots=True,
                                       get_snapshots=True,
                                       get_nn_field=True)
model.save()

# --------------------------------------------------------------------------------------------------------------------
# plot training

# plot training snapshots
model.plot_snapshots(snaps, title="01_train")

# plot neural field kernels
model.plot_nn_kernels(title="02_train")

# plot neural field
model.plot_neural_field(nn_field, title="03_train")

# plot expression neurons
model.plot_expression_neurons(expr_resp, title="04_train")

# --------------------------------------------------------------------------------------------------------------------
# predict model
# load data
val_data = load_data(config, train=False)
val_data[0] = segment_sequence(val_data[0], config['val_seg_start_idx'], config['seq_length'])
val_data[1] = segment_sequence(val_data[1], config['val_seg_start_idx'], config['seq_length'])
print("[LOAD] Shape val_data segmented", np.shape(val_data[0]))
print("[LOAD] Shape val_label segmented", np.shape(val_data[1]))

# predict model
expr_resp, snaps, nn_field = model.predict(val_data, get_snapshots=True, get_nn_field=True)

# --------------------------------------------------------------------------------------------------------------------
# plot testing

# plot training snapshots
model.plot_snapshots(snaps, title="01_test")

# plot neural field
model.plot_neural_field(nn_field, title="03_test")

# plot expression neurons
model.plot_expression_neurons(expr_resp, title="04_test", val=True)


np.save(os.path.join("models/saved/", config['config_name'], "preds_expr_neuron"), expr_resp)

expr_resp = np.load(os.path.join("models/saved/", config['config_name'], "preds_expr_neuron.npy"))
print("shape expr_neuron", np.shape(expr_resp))

# todo change this in the future to be taken care of directly in the load function!
# MATLAB script is cleaning the neutral frames prior the neural field -> therefore the neural field has less activity!
no_neutral_expr_resp = np.zeros(np.shape(expr_resp))
neutral_idx = np.array(config['val_neutral_frames_idx'])
for i in range(len(neutral_idx)):
    start = neutral_idx[i, 0]
    stop = neutral_idx[i, 1]
    no_neutral_expr_resp[i, start:stop] = expr_resp[i, start:stop]

# plot expression neurons in function of expressivity level
plot_expressivity_level(config, no_neutral_expr_resp,
                        title="05_test",
                        save_folder=os.path.join("models/saved/", config['config_name']),
                        show_legends=False)
