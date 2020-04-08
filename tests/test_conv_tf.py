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

kernel = np.zeros((5, 5, 3, 3))
kernel[2, 2, 1, 0] = 1
kernel[2, 2, 1, 1] = 2
print("shape kernel", np.shape(kernel))
kernel = np.moveaxis(kernel, 2, 0)
print("shape kernel", np.shape(kernel))
kernel = np.expand_dims(kernel, axis=4)
print("shape kernel", np.shape(kernel))
# kernel = np.repeat(kernel, 3, axis=4)
# print("shape kernel", np.shape(kernel))
print("---------------------------------------------------------")
print("kernel")
for i in range(3):
    print(np.squeeze(kernel[i, :, :, 0]))
    print(np.squeeze(kernel[i, :, :, 1]))
    print(np.squeeze(kernel[i, :, :, 2]))
    print()

outputs = []
for i in range(3):
    print("shape kernel", np.shape(kernel[i]))
    output = tf.nn.conv2d(input, kernel[i], strides=1, padding='SAME')
    print("shape output", np.shape(output))
    outputs.append(output)
print("---------------------------------------------------------")
print("shape outputs", np.shape(outputs))
outputs = np.squeeze(outputs)
print("shape outputs", np.shape(outputs))
print(outputs[0, :, :])
print(outputs[1, :, :])
print(outputs[2, :, :])
