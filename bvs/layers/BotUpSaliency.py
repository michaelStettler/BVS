import tensorflow as tf


class BotUpSaliency(tf.keras.layers.Layer):
    def __init__(self, ksize,
                 epsilon,
                 alphaX,
                 alphaY,
                 verbose=0):

        """

        :param ksize:
        :param epsilon:
        :param alphaX:
        :param alphaY:
        :param verbose: 0: no display
                        1: print output
                        2: save output image and kernels
                        3: print intermediate result
                        4: save intermediate image
        """
        super(BotUpSaliency, self).__init__()
        self.ksize = ksize  # kernel size

        self.W, self.J = self._build_interconnection()

    def build(self, input_shape):
        print("build")

    def call(self, input):
        print("call")

    def _build_interconnection(self):
        print("coucou")
        return -1, -1










