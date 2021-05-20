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
best_eye_brow_IoU_ft = [209, 148, 59, 208]

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

# keep only highest IoU semantic score
preds = preds[..., best_eye_brow_IoU_ft]
print("shape semantic feature selection", np.shape(preds))

# compute average feature maps
ft_average = np.mean(preds, axis=-1)
ft_average = np.expand_dims(ft_average, axis=3)
print("shape ft_average", np.shape(ft_average))

# concacenate feature maps for plotting
preds = np.concatenate((preds, ft_average), axis=-1)
print("shape concacenate preds", np.shape(preds))


# -------------------------------------------------------------------------------------------------------------------
# RBF

# build eyebrow template
# build template
eyebrow_mask = np.array([[7, 12], [8, 21]])
print("eyebrow_mask")
print(eyebrow_mask)
eyebrow_template = preds[0, eyebrow_mask[0, 0]:eyebrow_mask[0, 1], eyebrow_mask[1, 0]:eyebrow_mask[1, 1]]  # neutral
# eyebrow_template = preds[60, eyebrow_mask[0, 0]:eyebrow_mask[0, 1], eyebrow_mask[1, 0]:eyebrow_mask[1, 1]]  # max C2

# test mask
# preds[:, eyebrow_mask[0, 0]:eyebrow_mask[0, 1], eyebrow_mask[1, 0]:eyebrow_mask[1, 1], 0] = 5000
diff_template = preds[0, eyebrow_mask[0, 0]:eyebrow_mask[0, 1], eyebrow_mask[1, 0]:eyebrow_mask[1, 1]] - eyebrow_template
print("sum test diff_template", np.sum(diff_template))
print()

shape_pr = np.shape(preds)
print("shape shape_pr", shape_pr)
print("num dimensions", len(shape_pr))
shape_eb_temp = np.shape(eyebrow_template)
print("shape eyebrow_template", shape_eb_temp)

padd_x = shape_eb_temp[0] // 2
padd_y = shape_eb_temp[1] // 2
print("paddings", padd_x, padd_y)

# build padded prediction
padd_image = np.zeros((shape_pr[0], shape_pr[1] + shape_eb_temp[0] - 1, shape_pr[2] + shape_eb_temp[1] - 1, shape_pr[3]))
padd_image[:, padd_x:padd_x+shape_pr[1], padd_y:padd_y+shape_pr[2], :] = preds
print("shape padd_image", np.shape(padd_image))

# test template on padded image
print("x_start", padd_x+eyebrow_mask[0, 0], "x_stop", padd_x+eyebrow_mask[0, 1])
diff_template_padd = padd_image[0, padd_x+eyebrow_mask[0, 0]:padd_x+eyebrow_mask[0, 1], padd_y+eyebrow_mask[1, 0]:
                                                                                        padd_y+eyebrow_mask[1, 1]] - \
                     eyebrow_template
print("sum test diff_template_padd", np.sum(diff_template_padd))
print()

# convolve with the eyebrow template
diffs = []
for x in range(shape_pr[1]):
    for y in range(shape_pr[2]):
        patch = padd_image[:, x:x+shape_eb_temp[0], y:y+shape_eb_temp[1]]

        diffs.append(patch - np.repeat(np.expand_dims(eyebrow_template, axis=0), len(preds), axis=0))
print("shape diffs", np.shape(diffs))

# compute rbf kernel
sigma = 1.0
kernels = []
for diff in diffs:
    diff_ = np.reshape(diff, (len(diff), -1))  # flatten so we could compute the norm on axis 1
    kernels.append(np.exp(-np.linalg.norm(diff_, ord=2, axis=1) ** 2 / 2 / sigma ** 2))
print("shape kernels", np.shape(kernels))
kernels = np.moveaxis(kernels, -1, 0)
kernels = np.reshape(kernels, (shape_pr[0], shape_pr[1], shape_pr[2]))
print("shape kernels", np.shape(kernels))

eyebrow_ft = np.expand_dims(kernels, axis=3)
print("shape eyebrow_ft", np.shape(eyebrow_ft))

# concacenate feature maps for plotting
preds = np.concatenate((preds, eyebrow_ft), axis=-1)
print("shape concacenate preds", np.shape(preds))

# --------------------------------------------------------------------------------------------------------------------
# plots
plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
                "feature_maps_output.gif", verbose=True, video=True)

# plot dynamic
preds_ref = preds[0]
preds_dyn = preds - np.repeat(np.expand_dims(preds_ref, axis=0), len(preds), axis=0)
preds_dyn[preds_dyn < 0] = 0

plot_cnn_output(preds_dyn, os.path.join("models/saved", config["config_name"]),
                "dyn_feature_maps_output.gif", verbose=True, video=True)
