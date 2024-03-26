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

  @tf.function
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
    selected = tf.one_hot(selected,depth=self.num_mixtures)
    mu = tf.reduce_sum(selected*means,axis=-1)
    scale = tf.reduce_sum(selected*tf.exp(log_scales),axis=-1)

    #z = tfp.distributions.Logistic(loc=tf.zeros_like(mu),
    #                               scale=tf.ones_like(scale)).sample()
    z = tf.random.normal(shape=tf.shape(mu))
    samples = mu + z *scale

    samples = tf.clip_by_value(samples,-1,1)
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
      tf.concat([weights, means, log_scales], axis=-1)
    )
    return samples

  @tf.function
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
      inputs = tf.concat(stack,axis=1)
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

  def generate(self, length, condition=None, training=False):
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
      training (bool): Whether the model is training
    Returns:
      tf.Tensor: Output tensor"""
    if training:
      raise ValueError('This method should not be called during training.')
    batch_size = condition.shape[0] if condition is not None else 1
    if condition is None:
      condition = tf.zeros((1,1,2))
    input_shape= (batch_size,1,self._build_input_shape[0][2])
    sample = tf.random.normal(shape=input_shape)
    outputs = []

    # create queues
    for i,layer in zip(range(len(self.wavenet_layers)),self.wavenet_layers):
      for j in range(layer.kernel_size-1):
        # first layer is different because it is filled with noise
        # and first element is dequeued befor queued
        if i == 0:
          size = j + 1
          if not isinstance(self.qs[i][j],tf.queue.FIFOQueue):
            # if first run create queues
            self.qs[i][j] = tf.queue.FIFOQueue(
              size,
              tf.float32,
              (batch_size,1,self._build_input_shape[0][2])
            )
          else:
            # if not first run, clear queue
            enqueued = self.qs[i][j].size().numpy()
            self.qs[i][j].dequeue_many(enqueued)
          # initialize queue with noise
          self.qs[i][j].enqueue_many(
            tf.random.normal((size,batch_size,1,self._build_input_shape[0][2])))
        else:
          size = (j+1)*layer.dilation_rate
          if not isinstance(self.qs[i][j],tf.queue.FIFOQueue):
            # if first run create queues
            self.qs[i][j] = tf.queue.FIFOQueue(
              size,
              tf.float32,
              (batch_size,1,layer.channels)
            )
          else:
            # if not first run, clear queue
            enqueued = self.qs[i][j].size().numpy()
            self.qs[i][j].dequeue_many(enqueued)
          self.qs[i][j].enqueue_many(
            tf.zeros((size-1,batch_size,1,layer.channels)))

    for _ in tqdm(range(length),'Generating samples'):
      sample = self._generation(sample,condition)
      outputs.append(sample)
      for q in self.qs[0]:
        q.enqueue(sample)

    return tf.concat(outputs, axis=1)

  def generate_from_sample(self, length, sample, condition=None, training=False):
    if training:
      raise ValueError('This method should not be called during training.')
    batch_size = sample.shape[0]
    if condition is None:
      condition = tf.zeros((batch_size,1,2))
    elif condition.shape[0] != batch_size:
      raise ValueError('Condition must have the same batch size as sample.')

    cache = sample
    for i,layer in zip(range(len(self.wavenet_layers)),self.wavenet_layers):
      for j in range(layer.kernel_size-1):
        if i == 0:
          size = j + 1
          if not isinstance(self.qs[i][j],tf.queue.FIFOQueue):
            # if first run create queues
            self.qs[i][j] = tf.queue.FIFOQueue(
              size,
              tf.float32,
              (batch_size,1,self._build_input_shape[0][2])
            )
          else:
            # if not first run, clear queue
            enqueued = self.qs[i][j].size().numpy()
            self.qs[i][j].dequeue_many(enqueued)
          # initialize queue with sample
          # desired shape: (size,batch_size,1,self._build_input_shape[0][2])
          init = sample[:,-size-1:-2,:]
          init = tf.transpose(init, perm=[1,0,2])
          init = tf.expand_dims(init, axis=2)
          self.qs[i][j].enqueue_many(init)
        else:
          size = (j+1)*layer.dilation_rate
          if not isinstance(self.qs[i][j],tf.queue.FIFOQueue):
            # if first run create queues
            self.qs[i][j] = tf.queue.FIFOQueue(
              size,
              tf.float32,
              (batch_size,1,layer.channels)
            )
          else:
            # if not first run, clear queue
            enqueued = self.qs[i][j].size().numpy()
            self.qs[i][j].dequeue_many(enqueued)
          # initialize queue with cached operations
          # desired shape: (size-1,batch_size,1,layer.channels)
          init = cache[:,-size+1:,:]
          init = tf.transpose(init, perm=[1,0,2])
          init = tf.expand_dims(init, axis=2)
          self.qs[i][j].enqueue_many(init)
      # cache output of each layer
      cache = layer([cache,condition], training=training)

    outputs = [tf.expand_dims(x,axis=1) 
               for x in tf.unstack(sample, axis=1)]
    x = sample[:, -1:,:] # (batch,1,channels)
    for _ in tqdm(range(length),'Generating samples baser on sample'):
      x = self._generation(x,condition)
      outputs.append(x)
      for q in self.qs[0]:
        q.enqueue(x)

    return tf.concat(outputs, axis=1)

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
    