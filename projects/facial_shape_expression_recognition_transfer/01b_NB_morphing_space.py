import numpy as np
import os

from datasets_utils.morphing_space import transform_morph_space_list2space
from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase
from plots_utils.plot_morphing_space import plot_it_resp
from plots_utils.plot_morphing_space import plot_morphing_space

"""
test the naive PCA selection on the original morphing space dataset from the eLife paper

run: python -m projects.facial_shape_expression_recognition_transfer.01b_NB_morphing_space
"""

train_model = False
predict_it_resp = False

# load config
config_name = 'NB_morphing_space_m0001.json'
config = load_config(config_name, path='configs/norm_base_config')

# --------------------------------------------------------------------------------------------------------------------
# declare model
model = NormBase(config, input_shape=tuple(config['input_shape']), load_NB_model=True)

# train model
if train_model:
    print("[FIT] Train model]")
    data = load_data(config, train=True)
    model.fit(data,
              fit_dim_red=True,
              fit_ref=True,
              fit_tun=True)
    model.save()

# --------------------------------------------------------------------------------------------------------------------
# predict model
if predict_it_resp:
    data = load_data(config, train=False)
    print("[Test] Predict model")
    accuracy, it_resp, labels = model.evaluate(data)
    print("[Test] model predicted")
    np.save(os.path.join("models/saved", config['config_name'], "NormBase", "it_resp"), it_resp)
    print("[Test] it_resp saved!")
else:
    it_resp = np.load(os.path.join("models/saved", config['config_name'], "NormBase", "it_resp.npy"))
    print("[Test] it_resp loaded!")
print("[Test] shape it_resp", np.shape(it_resp))
print()

# --------------------------------------------------------------------------------------------------------------------
# plot

# reshape responses to a 25 stimuli x 150 frames x n_category
it_resp = np.reshape(it_resp, (25, 150, np.shape(it_resp)[-1]))
# remove the neutral responses
it_resp = it_resp[..., 1:]
print("[Plot] shape it_resp", np.shape(it_resp))

# plot it responses of the training prototypes
it_proto = it_resp[[20, 0, 24, 4]]  # keep only the 4 prototypes c1=20, c2=0, c3=24, c4=4
print("[Plot] shape it_proto", np.shape(it_proto))

plot_it_resp(it_proto, title="NB Human Prototypes",
             labels=config['train_expression'],
             save_folder=os.path.join(os.path.join("models/saved", config['config_name'])))

# plot the responses over the entire morphing space
# reshape the number of stimuli to a 5x5 grid
it_morph_space = transform_morph_space_list2space(it_resp)
print("[Plot] shape morphing space", np.shape(it_morph_space))
# it_morph_space = np.amax(it_morph_space, axis=2)  # compute max over all sequences
it_morph_space = np.sum(it_morph_space, axis=2)  # compute max over all sequences
print("[Plot] shape it_morph_space", np.shape(it_morph_space))
plot_morphing_space(it_morph_space,
                    save_folder=os.path.join(os.path.join("models/saved", config['config_name'])))
