"""WaveNet model without conditioning."""

import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm

from src.fastwavenet.layers import CondWaveNetLayer


class GlobCondWaveNet(tf.keras.Model):
  """WaveNet model with global conditioning."""

  def __init__(self, kernel_size, channels, layers, loss_fn,
               dilatation_bound=512, num_mixtures=10, bits=16,
               skip_channels=None, dilatation_channels=None,
               l2_reg_factor = None, **kwargs):
    """Initialize WaveNet model.

    Args:
      kernel_size (int): Kernel size for dilated convolutions
      channels (int): Number of channels in residual connections
      layers (int): Number of layers in WaveNet
      dilatation_bound (int): Maximum dilatation for layers
      num_mixtures (int): Number of mixtures in output distribution
      bits (int): Number of bits in input data
      skip_channels (int): Number of channels in skip connections
      dilatation_channels (int): Number of channels in dilatated conv
      l2_reg_factor (float): L2 regularization factor, if None or 0 no
                             regularization is used
    """
    super().__init__(**kwargs)

    #compute dilatations for layers
    if math.log(dilatation_bound,kernel_size) % 1 != 0:
      raise ValueError('Dilatation bound must be power of kernel_size.')
    max_power = math.log(dilatation_bound,kernel_size)+1
    dilatations = [int(kernel_size**(i % max_power))
                   for i in range(layers)]

    self.loss_fn = loss_fn(bits)

    self.wavenet_layers = [CondWaveNetLayer(dil, kernel_size, channels,
                                            dilatation_channels,
                                            skip_channels,
                                            l2_reg_factor=l2_reg_factor)
                   for dil in dilatations]
    if skip_channels is None:
      skip_channels = channels

    if l2_reg_factor is not None and l2_reg_factor > 0:
      r1 = tf.keras.regularizers.L2(l2_reg_factor)
      r2 = tf.keras.regularizers.L2(l2_reg_factor)
      self.regularization = True
    else:
      r1 = None
      r2 = None
      self.regularization = False

    self.pre_final = tf.keras.layers.Conv1D(kernel_size=1,
                                            filters=skip_channels,
                                            padding='same',
                                            kernel_regularizer=r1)
    self.final = tf.keras.layers.Conv1D(kernel_size=1,
                                        filters=3*num_mixtures,
                                        padding='same',
                                        kernel_regularizer=r2)

    # queues for fast generation
    self.qs = np.zeros((layers,kernel_size-1)).tolist()

    self.num_mixtures = num_mixtures

    self.receptive_field = 1
    for dil in dilatations:
      self.receptive_field += dil*(kernel_size-1)

  @tf.function(reduce_retracing=True)
  def call(self, inputs, training=False):
    """Call the model on input.

    Args:
      inputs (tuple): Input tensor and condition tensor
      training (bool): Whether the model is training
    Returns:
      tf.Tensor: Output tensor"""
    x, cond = inputs
    cond = tf.expand_dims(cond, axis=1)
    cond = tf.repeat(cond,repeats=x.shape[-1],axis=1)
    aggregate = None


    for layer in self.wavenet_layers:
      x, skip = layer([x,cond], training=training)

      if aggregate is None:
        aggregate = skip
      else:
        aggregate = aggregate + skip

    x = tf.keras.activations.relu(aggregate)
    x = self.pre_final(x)
    x = tf.keras.activations.relu(x)
    x = self.final(x)
    weights, means, log_scales = tf.split(x, 3, axis=-1)
    return weights, means, log_scales

  @tf.function(
    input_signature=[
      tf.TensorSpec(shape=[None, None], dtype=tf.float32), # Weights
      tf.TensorSpec(shape=[None, None], dtype=tf.float32), # Means
      tf.TensorSpec(shape=[None, None], dtype=tf.float32) # Log scales
    ]
  )
  def sample_from_output(self, weights, means, log_scales):
    """Sample from output distribution.

    Args:
      weights (tf.Tensor): Weights tensor
      means (tf.Tensor): Means tensor
      log_scales (tf.Tensor): Log scales tensor
    Returns:
      tf.Tensor: Sampled tensor"""
    weights = tf.math.log(weights+1e-10)

    selected = tf.random.categorical(weights,1)
    selected = tf.squeeze(selected,axis=-1)


    selected = tf.one_hot(selected,depth=self.num_mixtures)
    mu = tf.reduce_sum(selected*means,axis=-1)
    scale = tf.reduce_sum(selected*tf.exp(log_scales),axis=-1)

    #z = tfp.distributions.Logistic(loc=tf.zeros_like(mu),
    #                               scale=tf.ones_like(scale)).sample()

    z = tf.random.normal(shape=tf.shape(mu),seed=42)

    samples = mu + z *scale

    samples = tf.clip_by_value(samples,-1,1)
    samples = tf.expand_dims(samples,axis=-1)
    return samples

  @tf.function(
    input_signature=[
      tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # Weights
      tf.TensorSpec(shape=[None, None, None], dtype=tf.float32), # Means
      tf.TensorSpec(shape=[None, None, None], dtype=tf.float32) # Log scales
    ]
  )
  def sample_track(self, weights, means, log_scales):
    """Sample from output distribution. Takes a whole track as input.
    This means that the input tensors has shape (batch, length, mixtures).

    Args:
      weights (tf.Tensor): Weights tensor
      means (tf.Tensor): Means tensor
      log_scales (tf.Tensor): Log scales tensor
    Returns:
      tf.Tensor: Sampled tensor"""
    samples = tf.map_fn(
      lambda x: self.sample_from_output(*tf.split(x, 3, axis=-1)),
      tf.concat([weights, means, log_scales], axis=-1) # pylint: disable=E1123,E1120
    )
    return samples

  def _generation(self,x,cond):
    """Generate one sample from model.

    This function is based on fast-wavenet idea and implements
    generation with queues.

    Args:
      x (tf.Tensor): Input tensor of shape (batch,1,channels)
      cond (tf.Tensor): Condition tensor of shape (batch,encoding_size)

    Returns:
      tf.Tensor: Output tensor"""
    # prepare condition to correct format
    cond = tf.expand_dims(cond, axis=1)

    aggregate = 0
    for i,layer in enumerate(self.wavenet_layers):
      stack = [x]
      # get inputs from queues
      for q in self.qs[i]:
        stack.append(q.dequeue())

      stack.reverse()
      inputs = tf.concat(stack,axis=1) # pylint: disable=E1123,E1120
      x, skip = layer.generate((inputs,cond))

      # add outputs to queues
      if i < len(self.wavenet_layers)-1:
        for q in self.qs[i+1]:
          q.enqueue(x)

      aggregate = skip + aggregate

    x = tf.nn.relu(aggregate)
    kernel,bias = self.pre_final.weights
    x = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
    x = tf.nn.bias_add(x,bias)
    x = tf.nn.relu(x)
    kernel,bias = self.final.weights
    x = tf.nn.conv1d(x,kernel,stride=1,padding='VALID')
    x = tf.nn.bias_add(x,bias)

    sample = self.sample_track(*tf.split(x, 3, axis=-1))

    return sample

  @tf.function
  def _generation_no_queues(self, x, cond):
    """Generate one sample from model.

    This function is based on wavenet idea and implements
    generation without queues.

    Args:
      x (tf.Tensor): Input tensor of shape (batch,length,1)
      cond (tf.Tensor): Condition tensor of shape (batch,encoding_size)
    Returns:
      tf.Tensor: Output tensor"""
    weights, means, log_scales = self([x,cond], training=False)
    # sample from output
    sample = self.sample_from_output(weights[:,-1,:],
                                     means[:,-1,:],
                                     log_scales[:,-1,:])

    return tf.expand_dims(sample,axis=1)

  def generate(self, length, condition=None, use_queue=True, training=False):
    """Generate samples from model.

    This method is used for generating samples during inference and
    can be called directly. The method generates samples from random
    noise and returns the generated samples. It is not decorated with
    tf.function decorator to allow for dynamic length of generated
    samples and to speed up the first call.

    Args:
      length (int): Length of generated recordings
      condition (tf.Tensor): Condition on which to generate the data. Optional
        The condition is of shape batch x encoding_size. Batch size of generated
        data is infered from shape of condition tensor. If condition is not
        provided then batch size is 1.
      use_queue (bool): Whether to use queues for generation
      training (bool): Whether the model is training
    Returns:
      tf.Tensor: Output tensor"""
    if training:
      raise ValueError('This method should not be called during training.')
    batch_size = condition.shape[0] if condition is not None else 2
    if condition is None:
      condition = tf.zeros((batch_size,2))
    if not use_queue:
      input_shape= (batch_size,length,1)
      x = tf.random.normal(input_shape)
      outputs =  []
      for _ in tqdm(range(length),'Generating samples'):
        pred = self._generation_no_queues(x, condition)
        outputs.append(pred)
        x = tf.concat([x,pred],axis=1)[:,-length:,:] # pylint: disable=E1123,E1120

      return tf.concat(outputs, axis=1) # pylint: disable=E1123,E1120

    input_shape= (batch_size,1,1)
    sample = tf.random.normal(shape=input_shape)
    outputs = []

    # create queues
    channels = 1
    for i,layer in zip(range(len(self.wavenet_layers)),self.wavenet_layers):
      for j in range(layer.kernel_size-1):
        size = (j+1)*(layer.dilation_rate)
        self.qs[i][j] = tf.queue.FIFOQueue(
          size+1,
          tf.float32,
          (batch_size,1,channels)
        )
        # first layer is different because it is filled with noise
        if i == 0:
          # initialize queue with noise
          self.qs[i][j].enqueue_many(
            tf.random.normal((size,batch_size,1,channels)))
        else:
          # initialize queue with zeros
          self.qs[i][j].enqueue_many(
            tf.zeros((size,batch_size,1,channels)))
      channels = layer.channels

    gen_function = tf.function(self._generation)
    for _ in tqdm(range(length),'Generating samples'):
      for q in self.qs[0]:
        q.enqueue(sample)
      sample = gen_function(sample,condition)
      outputs.append(sample)


    return tf.concat(outputs, axis=1) # pylint: disable=E1123,E1120

  def generate_from_sample(self, length, sample,
                           condition=None, use_queue=True,
                           training=False):
    """Generate samples from model based on input sample.

    Args:
      length (int): Length of generated recordings
      sample (tf.Tensor): Sample on which to base the generation
      condition (tf.Tensor): Condition on which to generate the data. Optional
        The condition is of shape batch x encoding_size. Batch size of generated
        data is infered from shape of condition tensor. If condition is not
        provided then batch size is 1.
      use_queue (bool): Whether to use queues for generation
      training (bool): Whether the model is training"""
    if training:
      raise ValueError('This method should not be called during training.')
    batch_size = sample.shape[0]
    if condition is None:
      condition = tf.zeros((batch_size,2))
    elif condition.shape[0] != batch_size:
      raise ValueError('Condition must have the same batch size as sample.')
    if not use_queue:
      x = sample
      outputs =  []
      for _ in tqdm(range(length),'Generating samples based on sample'):
        pred = self._generation_no_queues(x,condition)
        outputs.append(pred)
        x = tf.concat([x,pred],axis=1)[:,-length:,:] # pylint: disable=E1123,E1120
      return tf.concat(outputs, axis=1) # pylint: disable=E1123,E1120

    # get last sample from input and remove it
    last = tf.expand_dims(sample[:,-1,:],axis=1)
    sample = sample[:,:-1,:]

    # pad sample if necessary
    if sample.shape[1] < self.receptive_field:
      sample = tf.pad(sample,[[0,0],
                              [self.receptive_field-sample.shape[1],0],
                              [0,0]])

    cache = sample
    channels = 1
    for i,layer in enumerate(self.wavenet_layers):
      for j in range(layer.kernel_size-1):
        size = (j+1)*(layer.dilation_rate)
        self.qs[i][j] = tf.queue.FIFOQueue(
          size+1,
          tf.float32,
          (batch_size,1,channels)
        )
        # initialize queue with sample
        # desired shape: (size,batch_size,1,channels)
        init = cache[:,-size:,:]
        init = tf.transpose(init, perm=[1,0,2])
        init = tf.expand_dims(init, axis=2)
        self.qs[i][j].enqueue_many(init)

      # cache output of each layer
      cond = tf.expand_dims(condition, axis=1)
      cache,_ = layer([cache,cond], training=training)
      channels = layer.channels

    outputs = []
    x = last # (batch,1,channels)
    gen_function = tf.function(self._generation)
    for _ in tqdm(range(length),'Generating samples based on sample'):
      for q in self.qs[0]:
        q.enqueue(x)
      x = gen_function(x,condition)
      outputs.append(x)


    return tf.concat(outputs, axis=1) # pylint: disable=E1123,E1120

  @tf.function
  def train_step(self, data):
    """Train the model on input data.

    Args:
      data (tuple): Tuple of input data and condtion. The input data
        should be of shape batch_size x length+1 x 1 and the condition
        of shape batch_size x encoding_size."""
    x, condition = data
    target = x[:, 1:,:]
    inputs = x[:, :-1,:]

    with tf.GradientTape() as tape:
      weights, means, log_scales = self((inputs,condition), training=True)
      loss = self.loss_fn(target, weights, means, log_scales)
      if self.regularization:
        regularization = tf.reduce_sum(self.losses)
        loss_final = loss + regularization
      else:
        loss_final = loss
    gradients = tape.gradient(loss_final, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    predictions = self.sample_track(weights, means, log_scales)

    self.compiled_metrics.update_state(target, predictions)
    out_dict = {m.name: m.result() for m in self.metrics}
    out_dict[self.loss_fn.name] = loss
    if self.regularization:
      out_dict['regularization_loss'] = regularization
    return out_dict

  def compute_receptive_field(self,sampling_frequency):
    """Compute the receptive field of the WaveNet model.

    Args:
      sampling_frequency (int): Sampling frequency of the model
    Returns:
      float: Receptive field in seconds"""
    return self.receptive_field/sampling_frequency
