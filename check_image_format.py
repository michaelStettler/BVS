"""
2021/01/21
This script checks the format of the dataset and its influence.
The documentation (https://keras.io/api/applications/vgg/) says:
    "Note: each Keras Application expects a specific kind of input preprocessing.
    For VGG19, call tf.keras.applications.vgg19.preprocess_input on your inputs before passing them to the model."
"""
import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.plot_cnn_output import plot_cnn_output

# parameter
config = load_config("norm_base_animate_cnn_response_t0001.json")
path ="models/saved/check_image_format"
if not os.path.exists(path): os.mkdir(path)

# load images
images,_ = load_data(config, train=config["dataset"])
image = images[0]

# check format, i expect RGB [0..255]
print("original shape:", image.shape)
print("original format:", image.dtype)
print("original value range:", [np.min(image), np.max(image)])
#original shape: (224, 224, 3)
#original format: float64
#original value range: [0.0, 237.0]

# create alternative format with tf.keras.applications.vgg19.preprocess_input
# format should then be RGB float32 [0..1]
# furthermore, imagenet average should be subtracted
image_pre = tf.keras.applications.vgg19.preprocess_input(np.copy(image))
print("new shape:", image_pre.shape)
print("new format:", image_pre.dtype)
print("new value range:", [np.min(image_pre), np.max(image_pre)])
#new shape: (224, 224, 3)
#new format: float64
#new value range: [-123.68, 116.061]


### CHECK OUTPUT ###
# create model
model = tf.keras.applications.VGG19(include_top=False, weights=config['weights'], input_shape=(224,224,3))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)

# check output original format
response = model.predict(np.array([image]))[0]
plot_cnn_output(response,path,"original.png",title="original image", image=image)

# check output preprocessed format
response_pre = model.predict(np.array([image_pre]))[0]
plot_cnn_output(response_pre,path,"preprocessed.png",title="preprocessed image", image=image_pre)

# compare response
diff = response - response_pre
print("norm of difference:", np.linalg.norm(diff))
print("range of difference:", [np.min(diff), np.max(diff)])
plot_cnn_output(diff,path,"difference_response.png")