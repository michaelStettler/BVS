import cv2
import json
from models.layers import draw_on_grid
from tests.v1_and_BotUpSaliency.utils import save_debugg_BotUp_output

from models.layers.layers import BotUpSaliency

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

"""
test Zhaoping's Li bottoom-up saliency model of V1 on different stimuli

run: python -m tests.t11_test_BotUp_saliency_layer
"""

np.set_printoptions(precision=4, linewidth=150)

print("Test BotUpSaliency Layer")
debug = False

# load example -> IMPORTANT! HERE THE INPUT IS DIRECTLY FED INTO THE LAYER
image_type = 'fig5.18G'
# load data
if image_type == 'code_example':
    img = get_code_example()
elif image_type == 'fig5.14F':
    img = get_fig_5_14F()
elif image_type == 'fig5.18A':
    img = get_fig_5_18A()
elif image_type == 'fig5.18B':
    img = get_fig_5_18B()
elif image_type == 'fig5.18D':
    img = get_fig_5_18D()
elif image_type == 'fig5.18G':
    img = get_fig_5_18G()
else:
    raise NotImplementedError("Please select a valid image_type to test!")

# load config
config = 'configs/simple_config.json'
with open(config) as f:
  config = json.load(f)

n_rot = config['n_rot']
print("Num orientations {}".format(config['n_rot']))

# build model
input = Input(shape=np.shape(img))

steps = 200
bu_saliency = BotUpSaliency((9, 9),
                            K=n_rot,
                            steps=steps,
                            epsilon=0.1,
                            verbose=0)

x = bu_saliency(input)


model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# predict image
test_img = np.expand_dims(img, axis=0)
pred = model.predict(x=test_img)

# save input
input_print = draw_on_grid(img)
print("shape input_print", np.shape(input_print))
print("min max input_print", np.min(input_print), np.max(input_print))
cv2.imwrite("layers/video/input.jpeg", input_print)

# plot saliency
if not debug:
    saliency = pred[0]
    print("shape saliency", np.shape(saliency))
    print("min max saliency", np.min(saliency), np.max(saliency))
    print(saliency)

    # normalize saliency
    saliency = saliency - np.min(saliency)
    saliency = saliency / np.max(saliency)
    saliency_map = np.expand_dims(saliency, axis=2)
    saliency_map = np.array(saliency_map * 255).astype(np.uint8)
    cv2.imwrite("layers/video/V1_saliency_map.jpeg", saliency_map.astype(np.uint8))


else:
    save_debugg_BotUp_output(pred, steps)



