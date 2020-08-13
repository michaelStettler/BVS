import tensorflow as tf
import numpy as np


class NormBase(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(NormBase, self).__init__()
        print("coucou")
        self.num_outputs = num_outputs

    def build(self, input_shape):
        print("build")
        self.m = tf.zeros([input_shape[-1]])
        print(self.m)
        self.n_mean = tf.zeros([n_outputs, input_shape[-1]])
        print(self.n_mean)
        self.n = tf.zeros([n_outputs, input_shape[-1]])
        self.n_cumul = tf.zeros([n_outputs])
        self.ref_cumul = 0

    def call(self, input):
        print("call")