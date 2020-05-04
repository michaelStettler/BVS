import tensorflow as tf
import numpy as np
from bvs.layers.BotUpSaliency import BotUpSaliency
from bvs.layers.GaborFilters import GaborFilters


class V1(tf.keras.layers.Layer):

    def __init__(self, ksize, K,
                 n_steps=32,
                 epsilon=0.01,
                 use_octave=True):

        """
        V1 layer uses gaborfilters and BotUP_saliency layers
        """
        super(V1, self).__init__()

        self.ksize = ksize
        self.K = K
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.use_octave = use_octave

    def build(self, input_shape):
        theta = np.array(range(self.K)) / self.K * np.pi
        self.contrast_gb = GaborFilters(self.ksize,
                                        theta=theta,
                                        octave=self.use_octave)

        self.contrast_BU = BotUpSaliency(self.ksize,
                                         K=self.K,
                                         steps=self.n_steps,
                                         epsilon=self.epsilon)

    def call(self, inputs, **kwargs):
        x = self.contrast_gb(inputs)
        # todo add BN or normalizes?
        x = x - tf.reduce_min(x)
        if tf.reduce_max != 0:
            x = x / tf.reduce_max(x)
        x = x * 1.5
        x = self.contrast_BU(x)
        x = tf.expand_dims(x, axis=3)
        x = tf.concat([inputs, x], axis=3)

        return x








