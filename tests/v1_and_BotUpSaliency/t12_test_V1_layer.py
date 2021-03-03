import cv2
import json

from models import V1

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

np.set_printoptions(precision=4, linewidth=150)

print("Test BotUpSaliency Layer")

# load example -> IMPORTANT! HERE THE INPUT IS DIRECTLY FED INTO THE LAYER
image_type = 'mnist'
# load data
if image_type == 'mnist':
    img = get_mnist(2)
else:
    raise NotImplementedError("Please select a valid image_type to test!")

# load config
config = 'configs/V1_simple_config.json'
with open(config) as f:
  config = json.load(f)

n_rot = config['n_rot']
n_steps = config['n_steps']
epsilon = config['epsilon']
print("[GB params] n_rot: {}".format(config['n_rot']))
print("[BU params] n_steps: {}, epsilon: {}".format(config['n_steps'], config['epsilon']))

# build model
input = Input(shape=np.shape(img))
x = V1((9, 9),
       K=n_rot,
       n_steps=n_steps,
       epsilon=epsilon)(input)
model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# predict image
test_img = np.expand_dims(img, axis=0)
pred = model.predict(x=test_img)

# save input
input_print = img.astype(np.uint8)
print("shape input_print", np.shape(input_print))
print("min max input_print", np.min(input_print), np.max(input_print))
cv2.imwrite("layers/video/input.jpeg", input_print)

# save saliency map
saliency = pred[0][:, :, -1]
print("shape saliency", np.shape(saliency))
print("min max saliency", np.min(saliency), np.max(saliency))
# print(np.squeeze(saliency))

# normalize saliency
saliency = saliency - np.min(saliency)
saliency = saliency / np.max(saliency)
saliency = np.array(saliency * 255).astype(np.uint8)
cv2.imwrite("layers/video/V1_saliency_map.jpeg", saliency)



