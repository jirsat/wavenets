import tensorflow as tf

class WaveNetLayer(tf.keras.layers.Layer):
    """WaveNet layer.
    
    This layer strictly follows paper and does not include conditioning.
    Some other public implementation include convolutions on residual connection,
    and different 1x1 convolutions for skip and main connections."""
    def __init__(self, dilation_rate, kernel, channels, **kwargs):
        super().__init__()
        self.dilated_conv = tf.keras.layers.Conv1D(
            filters=channels,
            kernel_size=kernel,
            dilation_rate=2*dilation_rate,
            padding='causal')
        self.conv1 = tf.keras.layers.Conv1D(
            filters=channels,
            kernel_size=1,
            padding='same')
        

    def call(self, inputs, training=False):
        x = inputs

        # create residual connection
        # this is place where some implementations use convolution
        # with kernel size 1 change number of channels
        residual = x

        # dilated convolution
        # some implementations use stack of dilated convolutions
        x = self.dilated_conv(x)

        # split into parts for gated activation
        # this allows usage of only one layer for both
        t,s = tf.split(x, 2, axis=-1)
        # gated activation
        x = tf.math.tanh(t) * tf.math.sigmoid(s)
        
        # 1x1 convolution
        x = self.conv1(x)

        # skip connection
        # some implementations use different 1x1 convolution here
        skip = x

        # add residual connection
        x = x + residual
        return x, skip


