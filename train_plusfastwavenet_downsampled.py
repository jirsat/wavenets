"""File for training the FastWaveNet model."""

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=2,--tf_xla_cpu_global_jit'

# pylint: disable=wrong-import-position
# import tensorflow after setting environment variables
import tensorflow as tf
from src.plusfastwavenet.non_cond_wavenet import NonCondWaveNet
from src.plusfastwavenet.loss import MixtureNormalLoss
from src.callbacks import UnconditionedSoundCallback, create_spectogram
from src.utils import train_test_split, preprocess_dataset
# pylint: enable=wrong-import-position


config = {
    'kernel_size': 4,
    'channels': 32,
    'layers': 12,
    'dilatation_bound': 1024,
    'batch_size': 32,
    'epochs': 1000,
    'lr': 0.0001,
    'recording_length': 8000,
    'num_mixtures': 10,
}
run_name = 'plusfastwavenet_8000'
preview_length = 8000 * 4

# Load data
dataset = tf.data.Dataset.load('./datasets/vctk8000')
FS = 8000

# take 1 male (59 ~ p286) and 1 female (4 ~ p229) speaker
# into test set and the rest into training set
test_speakers = [59, 4]
train_dataset, test_dataset = train_test_split(dataset, test_speakers)

# Preprocess data
train_dataset = preprocess_dataset(train_dataset, config['recording_length'],
                                   apply_mulaw=False,
                                   condition=False)
test_dataset = preprocess_dataset(test_dataset, config['recording_length'],
                                  apply_mulaw=False,
                                  condition=False)

train_dataset = train_dataset.shuffle(1000).batch(config['batch_size'])
test_dataset = test_dataset.batch(config['batch_size'])
example_batch = test_dataset.take(1).get_single_element()

# Create rebatch
model = NonCondWaveNet(config['kernel_size'], config['channels'],
                       config['layers'], MixtureNormalLoss,
                       config['dilatation_bound'])

# Compile model
callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath='./tmp/'+run_name,
    save_weights_only=True,
    monitor='mean_squared_error',
    mode='max',
    save_best_only=True),
  UnconditionedSoundCallback(
    './logs/'+run_name,
    frequency=FS,
    epoch_frequency=10,
    samples=preview_length,
    initial_sample=example_batch,
    apply_mulaw=False,
  ),
  tf.keras.callbacks.TensorBoard(log_dir='./logs/'+run_name,
                                 profile_batch=(10,15),
                                 write_graph=False),
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
samples = model.generate(preview_length,5)
tictoc = time.time()-tic
print(f'Generation took {tictoc}s')
print(f'Speed of generation was {preview_length/tictoc} samples/s')
