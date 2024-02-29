"""WaveNet model without conditioning."""

import tensorflow as tf
import math
import numpy as np

from src.wavenet.layers import WaveNetLayer


class NonCondWaveNet(tf.keras.Model):
  """WaveNet model without conditioning."""

  def __init__(self, kernel_size, channels, layers, dilatation_bound=512):
    super().__init__()

    #compute dilatations for layers
    assert math.log(dilatation_bound,kernel_size) % 1 == 0, "Dilatation bound must be power of kernel_size."
    dilatations = [int(kernel_size**(i%math.log(dilatation_bound,kernel_size))) for i in range(layers)]

    self.wavenet_layers = [WaveNetLayer(dil, kernel_size, channels) for dil in dilatations]
    self.pre_final = tf.keras.layers.Conv1D(kernel_size=1, filters=channels, padding='same')
    self.final = tf.keras.layers.Conv1D(kernel_size=1, filters=256, padding='same')
    self.softmax = tf.keras.layers.Softmax(axis=-1)
  
    self.discretization =  tf.keras.layers.Discretization(
      bin_boundaries=np.linspace(-1,1,256)[1:-1],
      output_mode='int',) # using int because one_hot doesn't work

    # TODO: compute receptive field
    self.receptive_field = 0
    for dil in dilatations:
      self.receptive_field += dil*(kernel_size-1)

  def call(self, inputs, training=False):
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
  def generate(self, length, batch_size=1, training=False):
    if training:
      raise ValueError("This method should not be called during training.")
    input_shape= (batch_size,*self._build_input_shape[1:])
    x = tf.random.normal(input_shape)
    outputs = []

    for _ in range(length):
      prediction = self(x, training=training)
      prediction = tf.random.categorical(tf.math.log(prediction[:,-1,:]), 1)
      
      sample = tf.gather(np.linspace(-1,1,256,dtype=np.float32),prediction)
      x = tf.concat([x[:,1:,:], sample], axis=1)
      outputs.append(sample)

    return tf.concat(outputs, axis=1)

  def train_step(self, data):
    target = data[:, 1:,0]
    target = self.discretization(target)
    target = tf.one_hot(target,256)
    inputs = data[:, :-1,:]

    with tf.GradientTape() as tape:
      predictions = self(inputs, training=True)
      loss = self.compiled_loss(target, predictions)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(target, predictions)
    return {m.name: m.result() for m in self.metrics}

  def compute_receptive_field(self,sampling_frequency):
    """Compute the receptive field of the WaveNet model."""
    return self.receptive_field/sampling_frequency
    