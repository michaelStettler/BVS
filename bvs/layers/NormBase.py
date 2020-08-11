import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np


class NormBase(tf.keras.layers.Layer):
    def __init__(self):
        print("coucou")

    def build(self, input_shape):
        print("build")

    def call(self, input):
        print("call")