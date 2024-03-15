"""Module for loss functions."""

import tensorflow as tf

class MixtureLoss():
  """Mixture loss function for PlusFastWaveNet.
  It computes negative log likelihood of the target given the mixture
  of logistic distributions defined by the weights, means and log_scales.
  """
  def __init__(self, bits, name: str = 'MixtureLoss', **kwargs):
    """Initialize MixtureLoss.

    Args:
      bits (int): Number of bits in input data
      name (str): Name of the loss function
    """
    del kwargs
    self.bits = bits
    self.name = name


  @tf.function
  def call(self, y_true, weights, means, log_scales, **kwargs):
    """Calculate loss.

    Args:
      y_true (tf.Tensor): Target tensor
      weights (tf.Tensor): Weights for mixtures
      means (tf.Tensor): Means for mixtures
      log_scales (tf.Tensor): Log scales for mixtures
    Returns:
      tf.Tensor: Loss tensor
    """
    del kwargs
    num_mixtures = means.shape[-1]
    target = tf.repeat(y_true, num_mixtures, axis=-1)
    weights = tf.nn.softmax(weights, axis=-1)
    halfbit = 0.5*1/(2**self.bits) # as ints are converted to floats
    log_scales = tf.maximum(log_scales, -7) # to avoid NaNs - as in PixelCNN++
    likelihood = tf.reduce_sum(
      weights*(tf.nn.sigmoid((target-means+halfbit)*tf.exp(-1.0*log_scales))
               - tf.nn.sigmoid((target-means-halfbit)*tf.exp(-1.0*log_scales))),
      axis=-1)
    loglikes = tf.math.log(likelihood)
    return tf.reduce_mean(-1.0 * loglikes)

  def __call__(self,*args):
    return self.call(*args)
