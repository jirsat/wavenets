"""Module for callbacks."""
import tensorflow as tf

class UnconditionedSoundCallback(tf.keras.callbacks.Callback):
  """Callback for saving generated sound"""
  def __init__(self, log_dir, frequency: int,
               epoch_frequency: int, samples: int):
    """Initialize callback

    The callback saves generated sound to tensorboard log directory
    at the end of each epoch. The generated sound is generated from
    random noise and condition. If no condition is provided, the model
    class handles it.

    Args:
      log_dir (str): Directory for saving logs
      frequency (int): Sample rate of the generated sound
      epoch_frequency (int): Frequency of saving sound
      samples (int): How many samples to generate.
        i.e. Desired length / sample rate
    """
    super().__init__()
    self.writer = tf.summary.create_file_writer(log_dir)
    self.frequency = frequency
    self.log_freq = epoch_frequency
    self.samples = samples

  def on_epoch_end(self, epoch, logs=None):
    """Save generated sound on epoch end"""
    if epoch % self.log_freq== self.log_freq-1:
      batch = self.model.generate(self.samples,
                                  batch_size=5)
      with self.writer.as_default():
        tf.summary.audio('generated',
             data=batch2tf_wav(batch),
             step=epoch,
             sample_rate=self.frequency,
             encoding='wav',
             max_outputs=5)

def batch2tf_wav(batch) -> tf.Tensor:
  """Function to convert batch from dataset or generator to tf.Tensor

  Args:
    batch (tf.Tensor or tuple): Batch of audio data
  Returns:
    tf.Tensor: Tensor with audio data
  """
  if isinstance(batch,tuple):
    batch = batch[1]
  batch = tf.expand_dims(batch, -1)
  return batch
