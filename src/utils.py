"""File for utily functions, such as dataset preparation."""
import tensorflow as tf

def train_test_split(dataset, test_speakers):
  """Split dataset into training and test set.

  Args:
    dataset (tf.data.Dataset): Dataset to split
    test_speakers (list): List of speakers to include in test set
  """
  @tf.function
  def filter_fn(x):
    speaker = x['speaker']
    outputs = tf.reduce_any(speaker == test_speakers)
    return outputs

  test_dataset = dataset.filter(filter_fn)
  train_dataset = dataset.filter(lambda x: not filter_fn(x))

  return train_dataset, test_dataset

def preprocess_dataset(dataset, recording_length, apply_mulaw, condition):
  """Preprocess dataset.

  Args:
    dataset (tf.data.Dataset): Dataset to preprocess
    recording_length (int): Length of the recordings
    apply_mulaw (bool): Whether to apply mu-law transformation
    condition (bool): Whether to create a condition
  """
  @tf.function(input_signature=[tf.TensorSpec(shape=(None,1),
                                              dtype=tf.float32)])
  def convert_and_split(x):
    if apply_mulaw:
      x = tf.sign(x) * (tf.math.log(1.0 + 255.0*tf.abs(x)) / tf.math.log(256.0))
    x = tf.signal.frame(x, axis=0,
                        frame_length=recording_length+1,
                        frame_step=recording_length)

    return x

  def preprocess(inputs):
    x = tf.cast(inputs['speech'],tf.float32)
    x = tf.expand_dims(x, axis=-1)
    x = convert_and_split(x)
    if condition:
      cond = tf.one_hot(inputs['gender'], 2)
      cond = tf.broadcast_to(cond, [tf.shape(x)[0],2])
      return (x, cond)
    return x

  def downscale(inputs):
    x = tf.cast(inputs['speech'],tf.float32)
    inputs['speech'] = x / 2**15
    return inputs

  if condition:
    def filter_fn(x,_):
      finite = tf.reduce_all(tf.math.is_finite(x))
      low_bound = tf.reduce_all(tf.math.greater_equal(x,-1))
      up_bound = tf.reduce_all(tf.math.less_equal(x,1))
      length = tf.shape(x)[0] == recording_length+1
      return finite & length & low_bound & up_bound
  else:
    def filter_fn(x):
      finite = tf.reduce_all(tf.math.is_finite(x))
      low_bound = tf.reduce_all(tf.math.greater_equal(x,-1))
      up_bound = tf.reduce_all(tf.math.less_equal(x,1))
      length = tf.shape(x)[0] == recording_length+1
      return finite & length & low_bound & up_bound

  # preprocess dataset
  if tf.math.reduce_max(dataset.take(1).get_single_element()['speech']) > 2:
    print('Seems like the dataset is not normalized correctly, ',
          'trying to normalize it to [-1,1] by dividing by 2^15.')
    dataset = dataset.map(downscale)
    print('New max value: ',
          tf.math.reduce_max(dataset.take(1).get_single_element()['speech']))
  dataset = dataset.map(preprocess).unbatch()


  # filter dataset to only recordings of correct length and finite values
  dataset = dataset.filter(filter_fn)

  return dataset
