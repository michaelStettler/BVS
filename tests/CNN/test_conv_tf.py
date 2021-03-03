import tensorflow as tf
import numpy as np

input = np.zeros((1, 5, 5, 3))
input[0, 1:4, 2, 0] = 1
print("---------------------------------------------------------")
print("input")
print(input[0, :, :, 0])
print(input[0, :, :, 1])
print(input[0, :, :, 2])
print()

# kernel = np.zeros((5, 5, 3, 3))
kernel = np.zeros((1, 1, 3, 3))
kernel[0, 0, 0, 1] = 1
kernel[0, 0, 1, 1] = 2
print("shape kernel", np.shape(kernel))
# kernel = np.repeat(kernel, 3, axis=4)
# print("shape kernel", np.shape(kernel))
print("---------------------------------------------------------")
print("kernel")
for i in range(3):
    print(kernel[:, :, 0, i])
    print(kernel[:, :, 1, i])
    print(kernel[:, :, 2, i])
    print()

outputs2 = tf.nn.conv2d(input, kernel, strides=1, padding='SAME')
print("shape outputs2", np.shape(outputs2))
# outputs2 = tf.nn.depthwise_conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME')
outputs2 = np.squeeze(outputs2)
print(outputs2[:, :, 0])
print(outputs2[:, :, 1])
print(outputs2[:, :, 2])


kernel = np.moveaxis(kernel, 3, 0)
print("shape kernel", np.shape(kernel))
kernel = np.expand_dims(kernel, axis=4)
print("shape kernel", np.shape(kernel))
outputs = []
for i in range(3):
    output = tf.nn.conv2d(input, kernel[i], strides=1, padding='SAME')
    outputs.append(output)

print("---------------------------------------------------------")
print("shape outputs", np.shape(outputs))
outputs = np.squeeze(outputs)
print("shape outputs", np.shape(outputs))
print(outputs[0, :, :])
print(outputs[1, :, :])
print(outputs[2, :, :])

