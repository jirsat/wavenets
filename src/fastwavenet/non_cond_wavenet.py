"""WaveNet model without conditioning."""

import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm

from src.fastwavenet.layers import WaveNetLayer


class NonCondWaveNet(tf.keras.Model):
  """WaveNet model without conditioning."""

  def __init__(self, kernel_size, channels, layers, dilatation_bound=512):
    """Initialize WaveNet model.

    Args:
      kernel_size (int): Kernel size for dilated convolutions
      channels (int): Number of channels in dilated convolutions
      layers (int): Number of layers in WaveNet
      dilatation_bound (int): Maximum dilatation for layers
    """
    super().__init__()

    #compute dilatations for layers
    if math.log(dilatation_bound,kernel_size) % 1 != 0:
      raise ValueError('Dilatation bound must be power of kernel_size.')

    max_power = math.log(dilatation_bound,kernel_size)+1
    dilatations = [int(kernel_size**(i % max_power))
                   for i in range(layers)]

    self.wavenet_layers = [WaveNetLayer(dil, kernel_size, channels)
                           for dil in dilatations]
    self.pre_final = tf.keras.layers.Conv1D(kernel_size=1,
                                            filters=channels,
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

    # TODO: check compute receptive field
    self.receptive_field = 0
    for dil in dilatations:
      self.receptive_field += dil*(kernel_size-1)

  @tf.function(reduce_retracing=True)
  def call(self, inputs, training=False):
    """Call the model on input.

    Args:
      inputs (tf.Tensor): Input tensor
      training (bool): Whether the model is training
    Returns:
      tf.Tensor: Output tensor"""
    x = inputs
    aggregate = None
    for layer in self.wavenet_layers:
      x, skip = layer(x, training=training)

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
  def _generation(self,x):
    """Generate one sample from model.

    This function is based on fast-wavenet idea and implements
    generation with queues.

    Args:
      x (tf.Tensor): Input tensor of shape (batch,1,channels)
    Returns:
      tf.Tensor: Output tensor"""
    i = 0
    aggregate = None
    for layer in self.wavenet_layers:
      stack = [x]
      # get inputs from queues
      for q in self.qs[i]:
        stack.append(q.dequeue())

      stack.reverse()
      inputs = tf.concat(stack,axis=1)
      x, skip = layer.generate(inputs)

      # add outputs to queues
      if i < len(self.wavenet_layers)-1:
        for q in self.qs[i+1]:
          q.enqueue(x)

      if aggregate is None:
        aggregate = skip
      else:
        aggregate = aggregate + skip
      i = i + 1

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


  def generate(self, length, batch_size=1, training=False):
    """Generate samples from model.
    
    This method is used for generating samples during inference and
    can be called directly. The method generates samples from random
    noise and returns the generated samples. It is not decorated with
    tf.function decorator to allow for dynamic length of generated
    samples and to speed up the first call.
    
    Args:
      length (int): Length of generated recordings
      batch_size (int): Number of recordings to generate
      training (bool): Whether the model is training
    Returns:
      tf.Tensor: Output tensor"""
    if training:
      raise ValueError('This method should not be called during training.')

    input_shape= (batch_size,1,self._build_input_shape[2])
    sample = tf.random.normal(input_shape)
    outputs = []

    # create queues
    for i,layer in zip(range(len(self.wavenet_layers)),self.wavenet_layers):
      for j in range(layer.kernel_size-1):
        if i == 0:
          size = j + 1
          self.qs[i][j] = tf.queue.FIFOQueue(
            size,
            tf.float32,
            (batch_size,1,self._build_input_shape[2])
          )
          self.qs[i][j].enqueue_many(
            tf.random.normal((size,batch_size,1,self._build_input_shape[2])))
        else:
          size = (j+1)*layer.dilation_rate
          self.qs[i][j] = tf.queue.FIFOQueue(
            size,
            tf.float32,
            (batch_size,1,layer.channels)
          )
          self.qs[i][j].enqueue_many(
            tf.zeros((size-1,batch_size,1,layer.channels)))

    for _ in tqdm(range(length)):
      sample = self._generation(sample)
      outputs.append(sample)
      for q in self.qs[0]:
        q.enqueue(sample)

    return tf.concat(outputs, axis=1)

  @tf.function
  def train_step(self, data):
    """Train the model on input data.
    
    Args:
      data (tf.Tensor): Input data which should be of shape
        batch_size x length+1 x 1"""
    target = data[:, 1:,0]
    target = self.discretization(target)
    inputs = data[:, :-1,:]

    with tf.GradientTape() as tape:
      predictions = self(inputs, training=True)
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
