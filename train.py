"""File for training of wavenets."""

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=1,--tf_xla_cpu_global_jit'

# pylint: disable=wrong-import-position
# import tensorflow after setting environment variables
import tensorflow as tf
import tensorflow_datasets as tfds
from src.model import WaveNet
from src.callbacks import ConditionedSoundCallback, create_spectogram
from src.utils import train_test_split, preprocess_dataset
# pylint: enable=wrong-import-position


config = {
  'epochs': 500,
  'lr': 0.001,
  'recording_length': 8000,
  'batch_size': 8,

  'kernel_size': 2,
  'channels': 32,
  'blocks': 10,
  'layers_per_block': 5,
  'activation': 'leaky_relu',
  'conditioning': 'global',
  'mapping_layers': [8,16,32],
  'mapping_activation': 'leaky_relu',
  'dropout': 0,
  'dilation_bound': 256,
  'num_mixtures': None,
  'sampling_function': 'categorical',
  'bits': 8,
  'skip_channels': None,
  'dilation_channels': None,
  'use_resiudal': True,
  'use_skip': True,
  'final_layers_channels': [256,256],
  'l2_reg_factor': 0.01,
}

run_name = 'rewrite_1_'
run_name += f'{(config["conditioning"])}_{config["sampling_function"]}_'
run_name += f'{config["recording_length"]}'
preview_length = 8000 * 4



# Load data
dataset = tf.data.Dataset.load('./datasets/vctk8000')
# dataset = tfds.load('vctk', split='train', shuffle_files=False,
#                     data_dir='./datasets/vctk')
FS = 8000

# take 1 male (59 ~ p286) and 1 female (4 ~ p229) speaker
# into test set and the rest into training set
test_speakers = [59, 4]

train_dataset, test_dataset = train_test_split(dataset, test_speakers)

# Preprocess data
train_dataset = preprocess_dataset(train_dataset, config['recording_length'],
                                   apply_mulaw=True, condition=True)
test_dataset = preprocess_dataset(test_dataset, config['recording_length'],
                                  apply_mulaw=True, condition=True)

train_dataset = train_dataset.shuffle(1000).batch(config['batch_size'])
test_dataset = test_dataset.batch(config['batch_size'])
example_batch,example_cond = train_dataset.take(1).get_single_element()

# filter the training dataset to be sure, that the length of the
# audio is as expected
@tf.function(
    input_signature=[tf.TensorSpec(shape=(None,None,1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,2), dtype=tf.float32)])
def filter_fn(x, y):
  return tf.shape(x)[1] == config['recording_length']+1
train_dataset = train_dataset.filter(filter_fn)



# Prepare for model compilation
callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath='./tmp/'+run_name+'.weights.h5',
    save_weights_only=True,
    monitor='mean_squared_error',
    mode='max',
    save_best_only=True),
  # ConditionedSoundCallback(
  #   './logs/'+run_name,
  #   frequency=FS,
  #   epoch_frequency=5,
  #   samples=preview_length,
  #   condition=example_cond,
  #   apply_mulaw=False,
  #   initial_sample=(example_batch,example_cond),
  # ),
  tf.keras.callbacks.TensorBoard(log_dir='./logs/'+run_name,
                                 profile_batch=(15,25),
                                 write_graph=False),
  tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error',
                                      factor=0.2,
                                      patience=5,
                                      min_lr=1e-6)
]

# Save example batch
print('Example batch:')
print(example_batch.shape)
print("Min: ", tf.math.reduce_min(example_batch))
print("Max: ", tf.math.reduce_max(example_batch))


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

# Create model
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
  model = WaveNet(kernel_size=config['kernel_size'],
                  channels=config['channels'],
                  blocks=config['blocks'],
                  layers_per_block=config['layers_per_block'],
                  activation=config['activation'],
                  conditioning=config['conditioning'],
                  mapping_layers=config['mapping_layers'],
                  mapping_activation=config['mapping_activation'],
                  dropout=config['dropout'],
                  dilation_bound=config['dilation_bound'],
                  num_mixtures=config['num_mixtures'],
                  sampling_function=config['sampling_function'],
                  bits=config['bits'],
                  skip_channels=config['skip_channels'],
                  dilation_channels=config['dilation_channels'],
                  use_residual=config['use_resiudal'],
                  use_skip=config['use_skip'],
                  final_layers_channels=config['final_layers_channels'],
                  l2_reg_factor=config['l2_reg_factor'])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr'],
                                                   clipnorm=1.0),
                metrics=[tf.keras.metrics.MeanSquaredError()],
                jit_compile=True,)

  # build the model
  model((example_batch[:,:-1],example_cond))

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
