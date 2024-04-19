"""Module for WaveNet layers."""
import tensorflow as tf

class WaveNetLayer(tf.keras.layers.Layer):
  """WaveNet layer.

  As the logic in layer for global and local conditioning is the same, this is
  something that should be managed in model; layer is agnostic in this regard.
  """
  def __init__(self, kernel =2,
               dilation_rate=1,
               activation=None,
               channels=32,
               residual=True,
               dilation_channels=None,
               skip_channels=None,
               l2_reg_factor=None,
               condition=False,
               dropout=0,
               **kwargs):
    """Initialize WaveNet layer.

    Args:
      kernel (int): Kernel size for convolutions
      dilation_rate (list or int): Dilation rate for causal convolutions,
        if list than create n dilated convolutions with different rates and
        only last is gated
      activation (tf.keras.activations): Activation function for the convolution
        layers (except gated activation), if only one convolution is used
        than this input is ignored. Default is None, meaning no activation
      channels (int): Number of channels in residual connection, output & input
      residual (bool): If to use residual connection, default is True
      dilation_channels (int): Number of channels in dillatated conv
        for gate and same for filter, if None it is set to channels
      skip_channels (int): Number of channels in skip connection,
        If None, than only one 1x1 convolution is used and skip connection
        is the same as main connection before adding residual connection.
      l2_reg_factor (float): L2 regularization factor for weights, default is
        None, meaning no regularization, same as 0
      condition (bool): If to use conditioning, default is False
      dropout (float): Dropout rate, default is 0, meaning no dropout
    """
    super().__init__(**kwargs)

    # prepare regularization
    reg = 0 if l2_reg_factor is None else l2_reg_factor

    # if dilation_channels is None, set it to channels
    if dilation_channels is None:
      dilation_channels = channels

    # if dilation_rate is list, create stack of dilated convolutions
    self.dilated_stack = []
    if isinstance(dilation_rate, list):
      for dil in dilation_rate[:-1]:
        self.dilated_stack.append(
          tf.keras.layers.Conv1D(
            filters=dilation_channels,
            kernel_size=kernel,
            dilation_rate=dil,
            padding='causal',
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.L2(reg)))
      # last dilation rate is for gated activation
      dilation_rate = dilation_rate[-1]

    # save activation for queues
    self.activation = activation

    # last dilation
    self.dilated_stack.append(
      tf.keras.layers.Conv1D(
        filters=2*dilation_channels,
        kernel_size=kernel,
        dilation_rate=dilation_rate,
        padding='causal',
        kernel_regularizer=tf.keras.regularizers.L2(reg)))


    # output convolution
    self.conv1 = tf.keras.layers.Conv1D(
      filters=channels,
      kernel_size=1,
      padding='same',
      kernel_regularizer=tf.keras.regularizers.L2(reg))

    if skip_channels is not None:
      # convolution for skip connection
      self.conv_skip = tf.keras.layers.Conv1D(
        filters=skip_channels,
        kernel_size=1,
        padding='same',
        kernel_regularizer=tf.keras.regularizers.L2(reg))
    else:
      # if skip_channels is None, use the output of the layer without res conn
      self.conv_skip = None

    if dropout > 0:
      self.dropout = tf.keras.layers.Dropout(dropout)
    else:
      self.dropout = None


    self.condition = condition
    if condition:
      self.conv_cond = tf.keras.layers.Conv1D(
        filters=2*dilation_channels,
        kernel_size=1,
        kernel_regularizer=tf.keras.regularizers.L2(reg))

    self.kernel_size = kernel
    self.dilation_rate = dilation_rate
    self.channels = channels
    self.residual = residual

  def build(self, input_shape):
    """Build method for WaveNet layer.

    Args:
      input_shape (tuple): Tuple of input shapes
    """
    if self.condition:
      # input is tuple of two tensors
      x_shape,cond_shape = input_shape
      # check if the condition tensor has the same length as the input
      if x_shape[1] != cond_shape[1]:
        raise ValueError('Condition tensor must have the same length as input')
    # input shape is (batch, samples, channels)
    if self.residual:
      res = x_shape
    if self.dropout is not None:
      self.dropout.build(x_shape)
    for dil_conv in self.dilated_stack:
      dil_conv.build(x_shape)
      x_shape = dil_conv.compute_output_shape(x_shape)
    if self.condition:
      self.conv_cond.build(cond_shape)
      cond_shape = self.conv_cond.compute_output_shape(cond_shape)
      if x_shape[1] != cond_shape[1]:
        raise ValueError('Condition tensor must have the same length as input')
    
    x_shape = (x_shape[0],x_shape[1],x_shape[2]//2)
    
    if self.conv_skip is not None:
      skip_shape = self.conv_skip.build(x_shape)
    else:
      skip_shape = x_shape

    self.conv1.build(x_shape)
    x_shape = self.conv1.compute_output_shape(x_shape)
    if self.residual:
      if x_shape != res:
        raise ValueError('Residual connection must have the same shape as input')
    self.built = True
    self.output_shape=(x_shape,skip_shape)

  def compute_output_shape(self, input_shape):
    """Compute output shape for WaveNet layer.

    Args:
      input_shape (tuple): Tuple of input shapes
    Returns:
      tuple: Tuple of output shapes
    """
    if not self.built:
      raise ValueError('Layer is not built')
    return self.output_shape

  def call(self, inputs, training = False):
    """Call method for WaveNet layer.

    Args:
      inputs (tf.Tensor): Input tensor
    Returns:
      tuple: Tuple of two tensors, main and skip connections
    """
    if self.condition:
      x, cond = inputs
    else:
      x = inputs

    # create residual connection if needed
    if self.residual:
      residual = x

    if self.dropout is not None:
      x = self.dropout(x)

    # dilated convolutions
    for dil_conv in self.dilated_stack:
      x = dil_conv(x)

    # add condition if needed
    if self.condition:
      x = x + self.conv_cond(cond)

    # split into parts for gated activation
    # this allows usage of only one layer for both
    t,s = tf.split(x, 2, axis=-1)
    # gated activation
    x = tf.math.tanh(t) * tf.math.sigmoid(s)

    # 1x1 convolution
    x_out = self.conv1(x)

    # skip connection
    if self.conv_skip is not None:
      skip = self.conv_skip(x)
    else:
      skip = x_out

    # add residual connection
    if self.residual:
      x_out = x_out + residual
    return x_out, skip

  @tf.function
  def generate(self,inputs):
    """Generate method for WaveNet layer.

    This implementation is for the fast-wavenet. This function doesn't provide
    a speedup for the original wavenet, but it should allow the use of
    the fast-wavenet.

    Args:
      inputs (tf.Tensor): Input tensor, must already include the dilation
        meaning that the input tensor should already be composed only of the
        samples that are needed for the dilation. Convolution here is done
        without dilation. If condition is used, the condition tensor should be
        included in the input tensor as a tupple.
    Returns:
      tuple: Tuple of two tensors, main and skip connections
    """
    if self.condition:
      x, cond = inputs
    else:
      x = inputs

    # create residual connection if needed
    # as the output is only one sample, we don't need to store the residual
    # connection in full
    if self.residual:
      residual = tf.expand_dims(x[:,-1,:],1)


    # instead of dilated convolution we use only normal convolution,
    # as the dilation is included in the input
    for dil_conv in self.dilated_stack:
      kernel,bias = dil_conv.weights
      x = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
      x = tf.nn.bias_add(x,bias)

    # add condition if needed
    if self.condition:
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
    if self.conv_skip is not None:
      kernel,bias = self.conv_skip.weights
      skip = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
      skip = tf.nn.bias_add(skip,bias)
    else:
      skip = x_out

    # add residual connection
    x_out = x_out + residual
    return x_out, skip
