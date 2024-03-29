"""WaveNet model without conditioning."""

import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm

from src.wavenet.layers import WaveNetLayer


class NonCondWaveNet(tf.keras.Model):
  """WaveNet model without conditioning."""

  def __init__(self, kernel_size, channels, layers, dilatation_bound=512,
               skip_channels=None, dilatation_channels=None, **kwargs):
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

    self.wavenet_layers = [WaveNetLayer(dil, kernel_size, channels,
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

    self.receptive_field = 1
    for dil in dilatations:
      self.receptive_field += dil*(kernel_size-1)

  @tf.function
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
  def _generate_one_sample(self,x, training=False):
    """Generate one sample from model.

    This method is used for generating samples during inference and should
    not be called directly. This function is decorated with tf.function
    decorator to speed up the computation.
    
    Args:
      x (tf.Tensor): Input tensor
      training (bool): Whether the model is training
    Returns:
      tf.Tensor: Output tensor"""
    prediction = self(x, training=training)
    prediction = tf.random.categorical(
      tf.math.log(prediction[:,-1,:]), 1)

    sample = tf.expand_dims(tf.gather(
      np.linspace(-1,1,256,dtype=np.float32),prediction),axis=-1)
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
    input_shape= (batch_size,*self._build_input_shape[1:])
    x = tf.random.normal(input_shape)
    outputs = []

    for _ in tqdm(range(length),'Generating samples'):
      sample = self._generate_one_sample(x)
      x = tf.concat([x[:,1:,:], sample], axis=1)
      outputs.append(sample)

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
