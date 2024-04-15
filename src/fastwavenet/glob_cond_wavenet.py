"""WaveNet model without conditioning."""

import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm

from src.fastwavenet.layers import CondWaveNetLayer


class GlobCondWaveNet(tf.keras.Model):
  """WaveNet model with global conditioning."""

  def __init__(self, kernel_size, channels, layers, dilatation_bound=512,
               dilatation_channels=None, skip_channels=None, **kwargs):
    """Initialize WaveNet model.

    Args:
      kernel_size (int): Kernel size for dilated convolutions
      channels (int): Number of channels in residual connections
      layers (int): Number of layers in WaveNet
      dilatation_bound (int): Maximum dilatation for layers
      dilatation_channels (int): Number of channels in dilatated conv
      skip_channels (int): Number of channels in skip connections
    """
    super().__init__(**kwargs)

    #compute dilatations for layers
    if math.log(dilatation_bound,kernel_size) % 1 != 0:
      raise ValueError('Dilatation bound must be power of kernel_size.')
    max_power = math.log(dilatation_bound,kernel_size)+1
    dilatations = [int(kernel_size**(i % max_power))
                   for i in range(layers)]

    self.wavenet_layers = [CondWaveNetLayer(dil, kernel_size, channels,
                                            dilatation_channels,
                                            skip_channels)
                   for dil in dilatations]
    if skip_channels is None:
      skip_channels = channels
    self.pre_final = tf.keras.layers.Conv1D(kernel_size=1,
                                            filters=skip_channels,
                                            padding='same')
    self.final = tf.keras.layers.Conv1D(kernel_size=1,
                                        filters=256,
                                        padding='same')
    self.softmax = tf.keras.layers.Softmax(axis=-1)

    self.discretization =  tf.keras.layers.Discretization(
      bin_boundaries=np.linspace(-1,1,256)[1:-1],
      output_mode='int',)

    # queues for fast generation
    self.qs = np.zeros((layers,kernel_size-1)).tolist()

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
    x = self.softmax(x)
    return x

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
    probs = tf.nn.softmax(x,-1)

    sample = tf.random.categorical(
      tf.math.log(probs[:,-1,:]), 1)
    sample = tf.expand_dims(tf.gather(
      np.linspace(-1,1,256,dtype=np.float32),sample),axis=-1)

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

    return tf.concat(outputs, axis=1) # pylint: disable=E1123,E1120

  @tf.function
  def train_step(self, data):
    """Train the model on input data.

    Args:
      data (tuple): Tuple of input data and condtion. The input data
        should be of shape batch_size x length+1 x 1 and the condition
        of shape batch_size x encoding_size."""
    x, condition = data
    target = self.discretization(x[:, 1:])
    inputs = x[:, :-1]

    with tf.GradientTape() as tape:
      predictions = self((inputs,condition), training=True)
      loss = self.compiled_loss(target, predictions)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(target, predictions)
    return {m.name: m.result() for m in self.metrics}

  def compute_receptive_field(self,sampling_frequency):
    """Compute the receptive field of the WaveNet model.

    Args:
      sampling_frequency (int): Sampling frequency of the model
    Returns:
      float: Receptive field in seconds"""
    return self.receptive_field/sampling_frequency
