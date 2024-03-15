"""File for training the FastWaveNet model."""

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=2,--tf_xla_cpu_global_jit'

# pylint: disable=wrong-import-position
# import tensorflow after setting environment variables
import tensorflow as tf
from src.plusfastwavenet.glob_cond_wavenet import GlobCondWaveNet
from src.callbacks import ConditionedSoundCallback, create_spectogram
from src.plusfastwavenet.loss import MixtureLoss
# pylint: enable=wrong-import-position


config = {
    'kernel_size': 2,
    'channels': 32,
    'dilatation_channels': 32,
    'skip_channels': 256,
    'layers': 50,
    'dilatation_bound': 512,
    'batch_size': 24,
    'epochs': 500,
    'lr': 0.001,
    'recording_length': 8000,
    'num_mixtures': 10,
    'l2_reg': 0.001,
}
run_name = 'globcond_plusfastwavenet_8000'
preview_length = 8000 * 4


# Load data
dataset = tf.data.Dataset.load('./datasets/vctk8000')
FS = 8000
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

  # prepare the condition
  condition = tf.one_hot(inputs['gender'], 2)
  condition = tf.broadcast_to(condition, [tf.shape(x)[0],2])

  return (x, condition)

train_dataset = train_dataset.map(preprocess).unbatch()
train_dataset = train_dataset.shuffle(1000).batch(config['batch_size'])
test_dataset = test_dataset.map(preprocess).rebatch(config['batch_size'])
example_batch,example_cond = train_dataset.take(1).get_single_element()

# Create model
model = GlobCondWaveNet(kernel_size=config['kernel_size'],
                        channels=config['channels'],
                        layers=config['layers'],
                        loss_fn=MixtureLoss,
                        dilatation_bound=config['dilatation_bound'],
                        skip_channels=config['skip_channels'],
                        dilatation_channels=config['dilatation_channels'],
                        num_mixtures=config['num_mixtures'],
                        l2_reg_factor=config['l2_reg'])

# Compile model
callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath='./tmp/'+run_name,
    save_weights_only=True,
    monitor='mean_squared_error',
    mode='max',
    save_best_only=True),
  ConditionedSoundCallback(
    './logs/'+run_name,
    frequency=FS,
    epoch_frequency=5,
    samples=preview_length,
    condition=example_cond,
    apply_mulaw=False,
  ),
  tf.keras.callbacks.TensorBoard(log_dir='./logs/'+run_name,
                                 profile_batch=(15,25),
                                 write_graph=False),
  tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error',
                                      factor=0.2,
                                      patience=5,
                                      min_lr=1e-6)
]

# Save example batch
print('Example batch shape:')
print(example_batch.shape)
spectogram = create_spectogram(example_batch, FS)
with tf.summary.create_file_writer('./logs/'+run_name).as_default():
  tf.summary.audio('original',
                   data=example_batch,
                   step=0,
                   sample_rate=FS,
                   encoding='wav',
                   max_outputs=5)
  tf.summary.image('original_spectogram',
                   data=spectogram,
                   step=0,
                   max_outputs=5)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
              metrics=[tf.keras.metrics.MeanSquaredError()])

# build the model
model.call((example_batch[:,:-1],example_cond))

# print receptive field
print('Receptive field')
print(model.receptive_field,' samples')
print(model.compute_receptive_field(FS),' seconds')

# Train model
model.fit(train_dataset, epochs=config['epochs'],
          callbacks=callbacks)

# Generate samples
tic = time.time()
samples = model.generate(preview_length,condition=example_cond)
tictoc = time.time()-tic
print(f'Generation took {tictoc}s')
print(f'Speed of generation was {preview_length/tictoc} samples/s')
