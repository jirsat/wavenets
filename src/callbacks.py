"""Module for callbacks."""
import tensorflow as tf

class SoundCallback(tf.keras.callbacks.Callback):
  """Callback for saving generated sound"""
  def __init__(self, log_dir,
               sampling_frequency: int,
               samples: int,
               apply_mulaw: bool,
               epoch_frequency: int = 1,
               condition: tf.Tensor = None,
               use_fast = False,
               initial_sample = None):
    """Initialize callback

    The callback saves generated sound to tensorboard log directory
    at the end of epoch. The generated sound is generated from
    random noise and condition. If no condition is provided, the model
    class handles it.

    Args:
      log_dir (str): Directory for saving logs
      sampling_frequency (int): Sample rate of the generated sound
      samples (int): How many samples to generate.
        i.e. Desired length / sample rate
      apply_mulaw (bool): Whether to apply mu-law transformation on generated
        sound
      epoch_frequency (int): After how many epochs to save sound
      condition (tf.Tensor): Condition for the model to generate sound from,
        if None, then assume no conditioning
      use_fast: Whether to generate sound with or
        without queue, if 'both', then use both (useful for debugging)
      initial_sample (tf.Tensor): Initial sample for the model
        to generate novel sample from. Defaults to None (only random noise),
        expected input is (waveform,condition)
    """
    super().__init__()
    if use_fast not in ['both',True,False]:
      raise ValueError('use_fast must be one of True, False, "both"')
    if epoch_frequency < 1:
      raise ValueError('epoch_frequency must be greater than 0')
    self.writer = tf.summary.create_file_writer(log_dir)
    self.sampling_frequency = sampling_frequency
    self.log_freq = epoch_frequency
    self.samples = samples
    self.condition = condition
    self.apply_mulaw = apply_mulaw
    self.initial_sample = initial_sample
    self.use_fast = use_fast

  def on_epoch_end(self, epoch, logs=None):
    """Save generated sound on epoch end"""
    del logs
    if epoch % self.log_freq != self.log_freq-1:
      return

    generated = {}
    if self.use_fast == 'both':
      batch_fast = self.model.generate(self.samples,
                                      batch_size=5,
                                      condition=self.condition,
                                      use_queues=True)
      generated['fast'] = batch_fast
      batch = self.model.generate(self.samples,
                                  batch_size=5,
                                  condition=self.condition,
                                  use_queues=False)
      generated['standard'] = batch
    else:
      batch = self.model.generate(self.samples,
                                  batch_size=5,
                                  condition=self.condition,
                                  use_queues=self.use_fast)
      generated['standard'] = batch
    if self.initial_sample is not None:
      if self.condition is not None:
        wave, cond = self.initial_sample
        wave = wave[:8,:,:]
        cond = cond[:8,:]
      else:
        wave = self.initial_sample[:8,:,:]
        cond = None
      if self.use_fast == 'both':
        initial = self.model.generate(self.samples,
                                      batch_size=5, # will be ignored
                                      condition=cond,
                                      sample=wave,
                                      use_queues=True)
        generated['with_initial_fast'] = initial
        initial = self.model.generate(self.samples,
                                      batch_size=5, # will be ignored
                                      condition=cond,
                                      sample=wave,
                                      use_queues=False)
        generated['with_initial'] = initial
      else:
        initial = self.model.generate(self.samples,
                                      batch_size=5, # will be ignored
                                      condition=cond,
                                      sample=wave,
                                      use_queues=self.use_fast)
        generated['with_initial'] = initial

    with self.writer.as_default():
      for key, batch in generated.items():
        if self.apply_mulaw:
          batch = inverse_mu_law(batch)
        spectrogram = create_spectrogram(batch,self.sampling_frequency)

        tf.summary.audio('generated_'+key,
                         data=batch,
                         step=epoch,
                         sample_rate=self.sampling_frequency,
                         encoding='wav',
                         max_outputs=8)
        tf.summary.image('generated_spectrogram_'+key,
                          data=spectrogram,
                          step=epoch,
                          max_outputs=8)

class AddLRToLogs(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    del epoch
    logs.update({'lr': self.model.optimizer.learning_rate.numpy()})

@tf.function(input_signature=[
  tf.TensorSpec(shape=[None,None,None],dtype=tf.float32)])
def inverse_mu_law(y: tf.Tensor):
  """Reverse mu-law transformation"""
  x = tf.sign(y)*(tf.pow(256.0,tf.abs(y))-1.0)/255.0
  return x

def create_spectrogram(data: tf.Tensor, sample_rate: int):
  """Create spectrogram from audio data

  Args:
    data (tf.Tensor): Audio data
    sample_rate (int): Sample rate of the audio data
  Returns:
    tf.Tensor: spectrogram of the audio data
  """
  del sample_rate
  data = tf.squeeze(data)
  spectrogram = tf.signal.stft(data,frame_length=256,
                              frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = tf.math.log(spectrogram+1e-5)
  spectrogram = tf.squeeze(spectrogram)
  spectrogram = tf.expand_dims(spectrogram,-1)

  # permute to match tensorboard expectations
  spectrogram = tf.transpose(spectrogram, perm=[0,2,1,3])

  # scale to 0-1
  min_val = tf.reduce_min(spectrogram)
  spectrogram = spectrogram - min_val
  max_val = tf.reduce_max(spectrogram)
  spectrogram = spectrogram / max_val
  return spectrogram
