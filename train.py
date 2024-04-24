"""File for training of wavenets."""

import os
import time
import argparse
import yaml

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=1,--tf_xla_cpu_global_jit'

# pylint: disable=wrong-import-position
# import tensorflow after setting environment variables
import tensorflow as tf
import tensorflow_datasets as tfds
from src.model import WaveNet
from src.callbacks import SoundCallback, create_spectogram, AddLRToLogs, inverse_mu_law
from src.utils import train_test_split, preprocess_dataset
# pylint: enable=wrong-import-position


config = {
  'epochs': 500,
  'lr': 0.0005,
  'recording_length': 8000,
  'batch_size': 64,
  'apply_mulaw': False,
  'jit_compile': False,

  'kernel_size': 2,
  'channels': 32,
  'blocks': 5,
  'layers_per_block': 5,
  'activation': 'leaky_relu',
  'conditioning': 'global', # global, local, None
  'mapping_layers': [8,16,32],
  'mapping_activation': 'leaky_relu',
  'dropout': 0.1,
  'dilation_bound': 256,
  'num_mixtures': 8,
  'sampling_function': 'gaussian',
  'bits': 16,
  'skip_channels': None,
  'dilation_channels': None,
  'use_resiudal': True,
  'use_skip': True,
  'final_layers_channels': [128,256],
  'l2_reg_factor': 0,
}

parser = argparse.ArgumentParser()
parser.add_argument('--configfile', type=str)
args = parser.parse_args()

if args.configfile is None:
  print('No config file provided, using default config')
else:
  with open(args.configfile) as f:
    config.update(yaml.safe_load(f))


run_name = args.configfile.split('/')[-1].split('.')[0]+'_'
run_name += f'{(config["conditioning"])}cond_{config["sampling_function"]}_'
run_name += f'{config["recording_length"]}'
preview_length = 8000 * 4

initial_epoch = 0
if os.path.exists('./logs/'+run_name):
  print('Run name already exists')
  try:
    checkpoints = os.listdir('./results/'+run_name)
  except FileNotFoundError:
    print('No checkpoints found')
    print('Directory:', run_name)
    exit('Rename or delete the old logs directory')
  checkpoints.sort()
  print('Checkpoints found:',checkpoints)
  print('Resuming from last checkpoint')
  checkpointname = checkpoints[-1]
  filename = checkpointname.split('.weights')[0]
  filename,learning_rate = filename.split('-lr')
  initial_epoch = int(filename.split('-e')[-1])
  print('Initial epoch: ',initial_epoch)
  print('Learning rate: ',learning_rate)
  config['lr'] = float(learning_rate)


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
                                   apply_mulaw=config['apply_mulaw'],
                                   condition=config['conditioning'] is not None)
test_dataset = preprocess_dataset(test_dataset, config['recording_length'],
                                  apply_mulaw=config['apply_mulaw'],
                                  condition=config['conditioning'] is not None)

train_dataset = train_dataset.shuffle(1000).batch(config['batch_size'])
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(config['batch_size'])
if config['conditioning'] is not None:
  example_batch, example_condition = train_dataset.take(1).get_single_element()
else:
  example_batch = train_dataset.take(1).get_single_element()


# Prepare for model compilation
if config['conditioning'] is not None:
  initial_sample = (example_batch, example_condition)
else:
  initial_sample = example_batch



callbacks = [
  AddLRToLogs(),
  tf.keras.callbacks.ModelCheckpoint(
    filepath='./results/'+run_name+'/weights-e{epoch}-lr{lr}.weights.h5',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True),
  SoundCallback(
    './logs/'+run_name,
    sampling_frequency=FS,
    epoch_frequency=5,
    samples=preview_length,
    condition=example_condition if config['conditioning'] is not None else None,
    apply_mulaw=False,
    initial_sample=initial_sample,
  ),
  tf.keras.callbacks.TensorBoard(log_dir='./logs/'+run_name,
                                 #profile_batch=(15,25),
                                 write_graph=False),
  tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                      factor=0.2,
                                      patience=5,
                                      min_lr=2e-8,
                                      min_delta=10,),
  tf.keras.callbacks.EarlyStopping(monitor='loss',
                                  patience=15,
                                  min_delta=10,
                                  restore_best_weights=True),
  tf.keras.callbacks.TerminateOnNaN(),
]

# Save example batch
print('Example batch:')
print(example_batch.shape)
print('Min: ', tf.math.reduce_min(example_batch))
print('Max: ', tf.math.reduce_max(example_batch))

if config['apply_mulaw']:
  sample_audio = inverse_mu_law(example_batch)
else:
  sample_audio = example_batch
spectogram = create_spectogram(sample_audio, FS)
with tf.summary.create_file_writer('./logs/'+run_name).as_default():
  tf.summary.audio('original',
                   data=sample_audio,
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
                jit_compile=config['jit_compile'],)

  # build the model
  #model((example_batch[:,:-1],example_cond))
  if config['conditioning'] is not None:
    model((example_batch[:,:-1],example_condition))
  else:
    model(example_batch[:,:-1])

  if 'checkpointname' in locals():
    model.load_weights('./results/'+run_name+'/'+checkpointname)

# print receptive field
print('Receptive field')
print(model.receptive_field,' samples')
print(model.compute_receptive_field(FS),' seconds')

# Train model
model.fit(train_dataset, epochs=config['epochs'],
          callbacks=callbacks,
          validation_data=test_dataset,
          initial_epoch=initial_epoch)

# Generate samples
tic = time.time()
samples = model.generate(
  preview_length,
  batch_size=config['batch_size'],
  condition=example_condition if config['conditioning'] is not None else None,
)
tictoc = time.time()-tic
print(f'Generation took {tictoc}s')
print(f'Speed of generation was {preview_length/tictoc} samples/s')

os.makedirs('./results/'+run_name+'/samples', exist_ok=True)
for i, sample in enumerate(samples):
  if config['apply_mulaw']:
    sample = inverse_mu_law(sample)
  tf.io.write_file(f'./results/{run_name}/samples/sample_{i}.wav',
                   tf.audio.encode_wav(sample, FS))
