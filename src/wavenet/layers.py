"""Module for WaveNet layers."""
import tensorflow as tf

class WaveNetLayer(tf.keras.layers.Layer):
  """WaveNet layer.

  This layer strictly follows paper and does not include conditioning.
  Some other public implementation include convolutions on residual connection.
  """
  def __init__(self, dilation_rate, kernel, channels,
               dilatation_channels = None, skip_channels = None,
               **kwargs):
    """Initialize WaveNet layer.

    Args:
      dilation_rate (int): Dilation rate for convolution
      kernel (int): Kernel size for convolution
      channels (int): Number of channels in residual connection
      dilatation_channels (int): Number of channels in dillatated conv
                                 for gate and same for filter, if None
                                  it is set to channels
      skip_channels (int): Number of channels in skip connection,
                           If None, than only one 1x1 convolution is used
                           and skip connection is the same as main connection
                           before adding residual connection
    """
    super().__init__(**kwargs)
    if dilatation_channels is None:
      dilatation_channels = channels
    self.dilated_conv = tf.keras.layers.Conv1D(
      filters=2*dilatation_channels,
      kernel_size=kernel,
      dilation_rate=dilation_rate,
      padding='causal')
    self.conv1 = tf.keras.layers.Conv1D(
      filters=channels,
      kernel_size=1,
      padding='same')

    if skip_channels is not None:
      self.conv_skip = tf.keras.layers.Conv1D(
        filters=skip_channels,
        kernel_size=1,
        padding='same')
    else:
      self.conv_skip = self.conv1

  @tf.function
  def call(self, inputs):
    """Call method for WaveNet layer.

    Args:
      inputs (tf.Tensor): Input tensor
    Returns:
      tuple: Tuple of two tensors, main and skip connections
    """
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
    x_out = self.conv1(x)

    # skip connection
    skip = self.conv_skip(x)

    # add residual connection
    x_out = x_out + residual
    return x_out, skip


class CondWaveNetLayer(tf.keras.layers.Layer):
  """Conditional WaveNet layer.

  This layer strictly follows paper and does include conditioning.
  As the logic in layer for global and local conditioning is the same, this is
  something that should be managed in model; layer is agnostic in this regard.
  Some other public implementation include convolutions on residual connection.
  """
  def __init__(self, dilation_rate, kernel, channels,
               dilatation_channels = None, skip_channels = None,
               **kwargs):
    """Initialize conditional WaveNet layer.

    Args:
      dilation_rate (int): Dilation rate for convolution
      kernel (int): Kernel size for convolution
      channels (int): Number of channels in residual convolution
      dilatation_channels (int): Number of channels in dillatated conv
                                 for gate and same for filter, if None
                                  it is set to channels
      skip_channels (int): Number of channels in skip connection,
                           If None, than only one 1x1 convolution is used
                           and skip connection is the same as main connection
                           before adding residual connection
    """
    super().__init__(**kwargs)
    if dilatation_channels is None:
      dilatation_channels = channels
    self.dilated_conv = tf.keras.layers.Conv1D(
      filters=2*dilatation_channels,
      kernel_size=kernel,
      dilation_rate=dilation_rate,
      padding='causal')
    self.conv_cond = tf.keras.layers.Conv1D(
      filters=2*dilatation_channels,
      kernel_size=1)
    self.conv1 = tf.keras.layers.Conv1D(
      filters=channels,
      kernel_size=1,
      padding='same')
    if skip_channels is not None:
      self.conv_skip = tf.keras.layers.Conv1D(
        filters=skip_channels,
        kernel_size=1,
        padding='same')
    else:
      self.conv_skip = self.conv1

  @tf.function
  def call(self, inputs):
    """
    Call method for conditional WaveNet layer.

    Args:
      inputs (tuple): Tuple of two tensors, input and condition
    Returns:
      tuple: Tuple of two tensors, main and skip connections
    """
    x, cond = inputs

    # create residual connection
    # this is place where some implementations use convolution
    # with kernel size 1 change number of channels
    residual = x

    # dilated convolution
    # some implementations use stack of dilated convolutions
    x = self.dilated_conv(x)

    # adding the condition
    x = x + self.conv_cond(cond)

    # split into parts for gated activation
    # this allows usage of only one layer for both
    t,s = tf.split(x, 2, axis=-1)
    # gated activation
    x = tf.math.tanh(t) * tf.math.sigmoid(s)

    # 1x1 convolution
    x_out = self.conv1(x)

    # skip connection
    skip = self.conv_skip(x)

    # add residual connection
    x_out = x_out + residual
    return x_out, skip


