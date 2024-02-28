"""File for training the WaveNet model."""
import tensorflow as tf
import tensorflow_datasets as tfds
from src.wavenet.non_cond_wavenet import NonCondWaveNet

config = {
    'kernel_size': 2,
    'channels': 32,
    'layers': 10,
    'dilatation_bound': 512,
    'batch_size': 32,
    'epochs': 10,
    'lr': 0.001,
    'recording_length': 16000,
}

# Load data
dataset = tfds.load('vctk', split='train', shuffle_files=False, data_dir='./datasets/vctk')
FS = 48000
BITS = 16

# take 1 male (59 ~ p286) and 1 female (4 ~ p229) speaker
# into test set and the rest into training set
test_speakers = [59, 4]
def filter_fn(x):
    return x['speaker_id'] in test_speakers
test_dataset = dataset.filter(filter_fn)
train_dataset = dataset.filter(lambda x: not filter_fn(x))

# Preprocess data
def preprocess(x):
    # as we are not using conditioning, we can just take the audio
    x = x['audio']

    # convert 16-bit integers to floats [-1, 1]
    x = (2* tf.cast(x, tf.float32) / (2.0**BITS)) - 1.0

    # apply the mu-law companding as in the original paper
    x = tf.sign(x) * (tf.math.log(1 + 255*tf.abs(x)) / tf.math.log(256))

    # add channel dimension
    x = tf.expand_dims(x, axis=-1)

    # cut the audio into chunks of length recording_length
    x = tf.image.extract_patches(x, sizes=[1, config['recording_length'], 1], strides=[1, config['recording_length'], 1], rates=[1, 1, 1], padding='valid')

    # the one-hot encoding is done in the training loop
    return x

train_dataset = train_dataset.map(preprocess).unbatch()
test_dataset = test_dataset.map(preprocess).unbatch()

"""
# Create model
model = NonCondWaveNet(config['kernel_size'], config['channels'], config['layers'], config['dilatation_bound'])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Train model
model.fit(train_dataset.batch(config['batch_size']), epochs=config['epochs'])
"""