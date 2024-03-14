"""Module for callbacks."""
import tensorflow as tf

class UnconditionedSoundCallback(tf.keras.callbacks.Callback):
  """Callback for saving generated sound"""
  def __init__(self, log_dir, frequency: int,
               epoch_frequency: int, samples: int,
               apply_mulaw: bool = True):
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
      apply_mulaw (bool): Whether to apply mu-law transformation, 
        default True for backwards compatibility
    """
    super().__init__()
    self.writer = tf.summary.create_file_writer(log_dir)
    self.frequency = frequency
    self.log_freq = epoch_frequency
    self.samples = samples
    self.apply_mulaw = apply_mulaw

  def on_epoch_end(self, epoch, logs=None):
    """Save generated sound on epoch end"""
    del logs
    if epoch % self.log_freq== self.log_freq-1:
      batch = self.model.generate(self.samples,
                                  batch_size=5)
      if self.apply_mulaw:
        batch = inverse_mu_law(batch)
      spectogram = create_spectogram(batch,self.frequency)
      with self.writer.as_default():
        tf.summary.audio('generated',
                         data=batch,
                         step=epoch,
                         sample_rate=self.frequency,
                         encoding='wav',
                         max_outputs=5)
        tf.summary.image('generated_spectogram',
                         data=spectogram,
                         step=epoch,
                         max_outputs=5)


class ConditionedSoundCallback(tf.keras.callbacks.Callback):
  """Callback for saving generated sound"""
  def __init__(self, log_dir, frequency: int,
               epoch_frequency: int, samples: int,
               condition: tf.Tensor, apply_mulaw: bool = True):
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
      condition (tf.Tensor): Condition for the model
      apply_mulaw (bool): Whether to apply mu-law transformation, 
        default True for backwards compatibility
    """
    super().__init__()
    self.writer = tf.summary.create_file_writer(log_dir)
    self.frequency = frequency
    self.log_freq = epoch_frequency
    self.samples = samples
    self.condition = condition
    self.apply_mulaw = apply_mulaw

  def on_epoch_end(self, epoch, logs=None):
    """Save generated sound on epoch end"""
    del logs
    if epoch % self.log_freq== self.log_freq-1:
      batch = self.model.generate(self.samples,
                                  condition=self.condition)
      if self.apply_mulaw:
        batch = inverse_mu_law(batch)
      spectogram = create_spectogram(batch,self.frequency)
      with self.writer.as_default():
        tf.summary.audio('generated',
                         data=batch,
                         step=epoch,
                         sample_rate=self.frequency,
                         encoding='wav',
                         max_outputs=5)
        tf.summary.image('generated_spectogram',
                          data=spectogram,
                          step=epoch,
                          max_outputs=5)


@tf.function(input_signature=[
  tf.TensorSpec(shape=[None,None,None],dtype=tf.float32)])
def inverse_mu_law(y: tf.Tensor):
  """Reverse mu-law transformation"""
  x = tf.sign(y)*(tf.pow(256.0,tf.abs(y))-1.0)/255.0
  return x

def create_spectogram(data: tf.Tensor, sample_rate: int):
  """Create spectogram from audio data
  
  Args:
    data (tf.Tensor): Audio data
    sample_rate (int): Sample rate of the audio data
  Returns:
    tf.Tensor: Spectogram of the audio data
  """
  del sample_rate
  data = tf.squeeze(data)
  spectogram = tf.signal.stft(data,frame_length=256,
                              frame_step=128)
  spectogram = tf.abs(spectogram)
  spectogram = tf.math.log(spectogram+1e-5)
  spectogram = tf.squeeze(spectogram)
  spectogram = tf.expand_dims(spectogram,-1)

  # permute to match tensorboard expectations
  spectogram = tf.transpose(spectogram, perm=[0,2,1,3])

  # scale to 0-1
  min_val = tf.reduce_min(spectogram)
  spectogram = spectogram - min_val
  max_val = tf.reduce_max(spectogram)
  spectogram = spectogram / max_val
  return spectogram
