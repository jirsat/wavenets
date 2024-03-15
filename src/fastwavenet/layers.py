"""Module for WaveNet layers."""
import tensorflow as tf

class WaveNetLayer(tf.keras.layers.Layer):
  """WaveNet layer.

  This layer strictly follows paper and does not include conditioning.
  Some other public implementation include convolutions on residual connection.
  """
  def __init__(self, dilation_rate, kernel, channels,
               dilatation_channels = None, skip_channels = None,
               l2_reg_factor = None, **kwargs):
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
      l2_reg_factor (float): L2 regularization factor for weights, default is 
                             None, meaning no regularization, same as 0

    """
    super().__init__(**kwargs)
    if l2_reg_factor is not None and l2_reg_factor > 0:
      r1 = tf.keras.regularizers.L2(l2_reg_factor)
      r2 = tf.keras.regularizers.L2(l2_reg_factor)
      r3 = tf.keras.regularizers.L2(l2_reg_factor)
    else:
      r1 = None
      r2 = None
      r3 = None

    if dilatation_channels is None:
      dilatation_channels = channels
    self.dilated_conv = tf.keras.layers.Conv1D(
      filters=2*dilatation_channels,
      kernel_size=kernel,
      dilation_rate=dilation_rate,
      padding='causal',
      kernel_regularizer=r1)
    self.conv1 = tf.keras.layers.Conv1D(
      filters=channels,
      kernel_size=1,
      padding='same',
      kernel_regularizer=r2)

    if skip_channels is not None:
      self.conv_skip = tf.keras.layers.Conv1D(
        filters=skip_channels,
        kernel_size=1,
        padding='same',
        kernel_regularizer=r3)
    else:
      self.conv_skip = self.conv1

    self.kernel_size = kernel
    self.dilation_rate = dilation_rate
    self.channels = channels

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

  @tf.function
  def generate(self,inputs):
    """Generate method for WaveNet layer.
    
    This implementation is for the fast-wavenet. This function doesn't provide 
    a speedupfor the original wavenet, but it should allow the use of
    the fast-wavenet.
    
    Args:
      inputs (tf.Tensor): Input tensor, must already include the dilatation
                          meaning that the input tensor should already be
                          composed only of the samples that are needed for
                          the dilatation. Convolution here is done without
                          dilatation. 
    Returns:
      tuple: Tuple of two tensors, main and skip connections
    """
    x = inputs

    # create residual connection
    # as the output is only one sample, we don't need to store the residual
    # connection in full
    residual = tf.expand_dims(x[:,-1,:],1)


    # instead of dilated convolution we use only normal convolution,
    # as the dilatation is included in the input
    kernel,bias = self.dilated_conv.weights
    x = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
    x = tf.nn.bias_add(x,bias)

    # split into parts for gated activation
    # this allows usage of only one layer for both
    t,s = tf.split(x, 2, axis=-1)
    # gated activation
    x = tf.math.tanh(t) * tf.math.sigmoid(s)

    # 1x1 convolution
    kernel,bias = self.conv1.weights
    x_out = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
    x_out = tf.nn.bias_add(x_out,bias)

    # skip connection
    kernel,bias = self.conv_skip.weights
    skip = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
    skip = tf.nn.bias_add(skip,bias)

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
               l2_reg_factor = None, **kwargs):
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
      l2_reg_factor (float): L2 regularization factor for weights, default is 
                             None, meaning no regularization, same as 0
    """
    super().__init__(**kwargs)
    if dilatation_channels is None:
      dilatation_channels = channels
    if l2_reg_factor is not None and l2_reg_factor > 0:
      r1 = tf.keras.regularizers.L2(l2_reg_factor)
      r2 = tf.keras.regularizers.L2(l2_reg_factor)
      r3 = tf.keras.regularizers.L2(l2_reg_factor)
      r4 = tf.keras.regularizers.L2(l2_reg_factor)
    else:
      r1 = None
      r2 = None
      r3 = None
      r4 = None

    self.dilated_conv = tf.keras.layers.Conv1D(
      filters=2*dilatation_channels,
      kernel_size=kernel,
      dilation_rate=dilation_rate,
      padding='causal',
      kernel_regularizer=r1)
    self.conv_cond = tf.keras.layers.Conv1D(
      filters=2*dilatation_channels,
      kernel_size=1,
      kernel_regularizer=r2)
    self.conv1 = tf.keras.layers.Conv1D(
      filters=channels,
      kernel_size=1,
      padding='same',
      kernel_regularizer=r3)
    if skip_channels is not None:
      self.conv_skip = tf.keras.layers.Conv1D(
        filters=skip_channels,
        kernel_size=1,
        padding='same',
        kernel_regularizer=r4)
    else:
      self.conv_skip = self.conv1

    self.kernel_size = kernel
    self.dilation_rate = dilation_rate
    self.channels = channels

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

  @tf.function
  def generate(self,inputs):
    """Generate method for WaveNet layer.
    
    This implementation is for the fast-wavenet. This function doesn't provide 
    a speedupfor the original wavenet, but it should allow the use of the 
    fast-wavenet.
    
    Args:
      inputs (tf.Tensor,tf.Tenosor): Input tensor, must already include the 
      dilatation and condition
    Returns:
      tuple: Tuple of two tensors, main and skip connections
    """
    x, cond = inputs

    # create residual connection
    # as the output is only one sample, we don't need to store the residual
    # connection in full
    residual = tf.expand_dims(x[:,-1,:],1)


    # instead of dilated convolution we use only normal convolution,
    # as the dilatation is included in the input
    kernel,bias = self.dilated_conv.weights
    x = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
    x = tf.nn.bias_add(x,bias)

    # adding the condition
    kernel,bias = self.conv_cond.weights
    cond = tf.nn.conv1d(cond,kernel,stride=1,padding='VALID')
    cond = tf.nn.bias_add(cond,bias)
    x = x + cond

    # split into parts for gated activation
    # this allows usage of only one layer for both
    t,s = tf.split(x, 2, axis=-1)
    # gated activation
    x = tf.math.tanh(t) * tf.math.sigmoid(s)

    # 1x1 convolution
    kernel,bias = self.conv1.weights
    x_out = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
    x_out = tf.nn.bias_add(x_out,bias)

    # skip connection
    kernel,bias = self.conv_skip.weights
    skip = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
    skip = tf.nn.bias_add(skip,bias)

    # add residual connection
    x_out = x_out + residual
    return x_out, skip
