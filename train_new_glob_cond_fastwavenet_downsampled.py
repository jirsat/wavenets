"""File for training the FastWaveNet model."""

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=2,--tf_xla_cpu_global_jit'

# pylint: disable=wrong-import-position
# import tensorflow after setting environment variables
import tensorflow as tf
from src.fastwavenet.glob_cond_wavenet import GlobCondWaveNet
from src.callbacks import ConditionedSoundCallback, inverse_mu_law, create_spectogram
from src.utils import train_test_split, preprocess_dataset
# pylint: enable=wrong-import-position


config = {
    'kernel_size': 2,
    'channels': 32,
    'dilatation_channels': 32,
    'skip_channels': 512,
    'layers': 50,
    'dilatation_bound': 512,
    'batch_size': 32,
    'epochs': 1000,
    'lr': 0.0001,
    'recording_length': 8000,
}


run_name = 'globcond_fastwavenet_8000'
preview_length = 8000 * 4


# Load data
dataset = tf.data.Dataset.load('./datasets/vctk8000')
FS = 8000
BITS = 16

# take 1 male (59 ~ p286) and 1 female (4 ~ p229) speaker
# into test set and the rest into training set
test_speakers = [59, 4]

train_dataset, test_dataset = train_test_split(dataset, test_speakers)

# Preprocess data
train_dataset = preprocess_dataset(train_dataset, config['recording_length'],
                                   apply_mulaw=False,
                                   condition=True)
test_dataset = preprocess_dataset(test_dataset, config['recording_length'],
                                  apply_mulaw=False,
                                  condition=True)

train_dataset = train_dataset.shuffle(1000).batch(config['batch_size'])
test_dataset = test_dataset.batch(config['batch_size'])
example_batch,example_cond = train_dataset.take(1).get_single_element()

# Create model
model = GlobCondWaveNet(config['kernel_size'], config['channels'],
                       config['layers'], config['dilatation_bound'],
                       config['dilatation_channels'], config['skip_channels'])

# Compile model
callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath='./tmp/'+run_name,
    save_weights_only=True,
    monitor='sparse_categorical_accuracy',
    mode='max',
    save_best_only=True),
  ConditionedSoundCallback(
    './logs/'+run_name,
    frequency=FS,
    epoch_frequency=10,
    samples=preview_length,
    condition=example_cond
  ),
  tf.keras.callbacks.TensorBoard(log_dir='./logs/'+run_name,
                                 profile_batch=(10,15),
                                 write_graph=False),
]

# Save example batch
print('Example batch shape:')
print(example_batch.shape)
spectogram = create_spectogram(inverse_mu_law(example_batch),FS)
with tf.summary.create_file_writer('./logs/'+run_name).as_default():
  tf.summary.audio('original',
                   data=inverse_mu_law(example_batch),
                   step=0,
                   sample_rate=FS,
                   encoding='wav',
                   max_outputs=5)
  tf.summary.image('original_spectogram',
                   data=spectogram,
                   step=0,
                   max_outputs=5)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

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
