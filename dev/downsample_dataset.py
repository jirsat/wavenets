"""This script is used to downsample the dataset to a lower sampling rate."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import scipy

# run eagerly
tf.config.run_functions_eagerly(True)


# load the original dataset
dataset = tfds.load('vctk', split='train', shuffle_files=False,
                    data_dir='./datasets/vctk')
FS = 48000
BITS = 16

# set desired properties
new_fs = 8000
save_dir = './datasets/vctk8000'

# compute the downasmpling factor
downsample_factor = FS // new_fs
assert FS % new_fs == 0, 'The new sampling rate must be a divisor of the original one'

# prepare the conversion function
@tf.py_function(Tout=tf.float32)
def resampler(x,remove_n_samples,new_samples):
  x = x.numpy()
  if remove_n_samples.numpy()>0:
    x = x[:-remove_n_samples.numpy()]

  outputs = scipy.signal.resample(x,new_samples.numpy())
  return outputs

def convert_and_downsample(inputs):
  # select only audio for resampling
  x = tf.cast(inputs['speech'],tf.float32)

  # convert to -1 to 1 float
  x = (x / (2.0**(BITS-1)))

  length = tf.shape(x)[0]
  
  # number of samples to remove to have new samples aligned
  remove_n_samples = length%downsample_factor
  
  # compute new number of samples
  new_samples = int((length-remove_n_samples)/downsample_factor) 
  
  #resample the signal
  x = resampler(x,remove_n_samples,new_samples)

  # ensure the numpy is converted to tensor
  inputs['speech'] = tf.cast(x,tf.float32)
  return inputs

test_data = dataset.take(1)

print('Before processing')
print(test_data.get_single_element())
print('After processing')
print(test_data.map(
  convert_and_downsample).get_single_element())

# apply the conversion function on dataset
dataset = dataset.map(convert_and_downsample,
                      num_parallel_calls=tf.data.AUTOTUNE,
                      deterministic=False)
dataset = dataset.filter(lambda x:
  tf.shape(x['speech'])[0]>1)

print('Start saving')
dataset.save(save_dir)