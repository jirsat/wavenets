"""File for training the WaveNet model."""

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=2,--tf_xla_cpu_global_jit'

# pylint: disable=wrong-import-position
# import tensorflow after setting environment variables
import tensorflow as tf
import tensorflow_datasets as tfds
from src.wavenet.non_cond_wavenet import NonCondWaveNet
from src.callbacks import UnconditionedSoundCallback
# pylint: enable=wrong-import-position

config = {
    'kernel_size': 2,
    'channels': 32,
    'layers': 6,
    'dilatation_bound': 512,
    'batch_size': 24,
    'epochs': 1,
    'lr': 0.001,
    'recording_length': 24000,
}

# Load data
dataset = tfds.load('vctk', split='train', shuffle_files=False,
                    data_dir='./datasets/vctk')
FS = 48000
BITS = 16

# take 1 male (59 ~ p286) and 1 female (4 ~ p229) speaker
# into test set and the rest into training set
test_speakers = [59, 4]

@tf.function
def filter_fn(x):
  speaker = x['speaker']
  outputs = tf.reduce_any(speaker == test_speakers)
  return outputs

test_dataset = dataset.filter(filter_fn)
train_dataset = dataset.filter(lambda x: not filter_fn(x))

# Preprocess data
@tf.function(input_signature=[tf.TensorSpec(shape=(None,1), dtype=tf.float32)])
def convert_and_split(x):
  # convert 16bit integers (passed as floats) to floats [-1, 1]
  # the integers are from -2^15 to 2^15-1, therefore we ony need to divide them
  x = (x / (2.0**(BITS-1)))

  # apply the mu-law as in the original paper
  x = tf.sign(x) * (tf.math.log(1.0 + 255.0*tf.abs(x)) / tf.math.log(256.0))

  # split into chunks of size config['recording_lenght']
  x = tf.signal.frame(x, axis=0,
                      frame_length=config['recording_length']+1,
                      frame_step=config['recording_length'])

  return x

def preprocess(inputs):
  # as we are not using conditioning, we can just take the audio
  x = tf.cast(inputs['speech'],tf.float32)

  # add channel dimension
  x = tf.expand_dims(x, axis=-1)

  # cut the audio into chunks of length recording_length
  x = convert_and_split(x)

  # the one-hot encoding is done in the training loopd
  return x

train_dataset = train_dataset.map(preprocess).unbatch()
train_dataset = train_dataset.shuffle(1000).batch(config['batch_size'])
test_dataset = test_dataset.map(preprocess).rebatch(config['batch_size'])
example_batch = train_dataset.take(1).get_single_element()

# Create model
model = NonCondWaveNet(config['kernel_size'], config['channels'],
                       config['layers'], config['dilatation_bound'])

# Compile model
callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath='./tmp/uncond_wavenet',
    save_weights_only=False,
    monitor='categorical_accuracy',
    mode='max',
    save_best_only=True),
  UnconditionedSoundCallback(
    './logs/wavenet',
    frequency=FS,
    epoch_frequency=10,
    samples=config['recording_length']
  )
]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

# build the model
model.call(example_batch[:,:-1])

# print receptive field
print('Receptive field')
print(model.receptive_field,' samples')
print(model.compute_receptive_field(FS),' seconds')

# Train model
model.fit(train_dataset, epochs=config['epochs'],
          callbacks=callbacks)

# Generate samples
tic = time.time()
samples = model.generate(config['recording_length'])
tictoc = tic-time.time()
print(f'Generation took {tictoc}s')
print(f'Speed of generation was {config["recording_length"]/tictoc} samples/s')
