"""Wavenet model"""

import math
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from src.layers import WaveNetLayer

SQRT2PI = tf.sqrt(2.0*3.14159265359)

class WaveNet(tf.keras.Model):
  """WaveNet model class"""

  def __init__(self,
               kernel_size: int = 2,
               channels: int = 32,
               blocks: int = 10,
               layers_per_block: int = 1,
               activation = None,
               conditioning = None,
               mapping_layers = None,
               mapping_activation = None,
               dropout: float = 0,
               dilation_bound: int = 512,
               num_mixtures = None,
               sampling_function: str = 'categorical',
               bits=8,
               skip_channels=None,
               dilation_channels=None,
               use_residual=True,
               use_skip=True,
               final_layers_channels=None,
               l2_reg_factor: int = 0,
               **kwargs):
    """Init function for WaveNet model class

    Args:
        kernel_size (int): kernel size for the convolutional layers
        channels (int): number of channels in the convolutional layers
         (int): number of layers in the model
        loss_fn (tf.keras.losses): loss function for the model
        dilation_bound (int): maximum dilation in the model
        num_mixtures (int): number of mixtures in the model
        bits (int): number of bits in the audio
        skip_channels (int): number of channels in the skip connections
        dilation_channels (int): number of channels in the dilation
        l2_reg_factor (float): l2 regularization factor
    """
    super(WaveNet, self).__init__(**kwargs)

    # check for valid input
    if conditioning not in ['global', 'local', None]:
      raise ValueError("Conditioning must be 'global', 'local' or None.")
    if kernel_size < 2:
      raise ValueError('Kernel size must be at least 2.')
    if math.log(dilation_bound,kernel_size) % 1 != 0:
      raise ValueError('dilation bound must be power of kernel_size.')
    if layers_per_block < 1:
      raise ValueError('Layers per block must be at least 1.')
    if blocks < 1:
      raise ValueError('Blocks must be at least 1.')
    if num_mixtures is not None and num_mixtures < 1:
      raise ValueError('Number of mixtures must be at least 1 or None.')
    if dropout < 0 or dropout > 1:
      raise ValueError('Dropout must be between 0 and 1.')
    if sampling_function not in ['categorical', 'logistic','gaussian']:
      raise ValueError('Sampling function must be categorical, '+
        'logistic or gaussian.')
    if sampling_function == 'categorical' and num_mixtures is not None:
      raise ValueError('Categorical sampling cannot be used with mixtures.')

    self.regularization = l2_reg_factor>0
    self.num_mixtures = num_mixtures
    self.use_skip = use_skip
    self.sampling_function = sampling_function
    self.bits = bits

    # create stack of dilated convolutions blocks
    max_power = int(math.log(dilation_bound, kernel_size))
    dilations = [kernel_size**(i % max_power)
                   for i in range(layers_per_block*blocks)]


    self.causal = tf.keras.layers.Conv1D(
      filters=channels,
      kernel_size=kernel_size,
      padding='causal',
      kernel_regularizer=tf.keras.regularizers.L2(l2_reg_factor))
    self.wavenet_blocks = [
      WaveNetLayer(
        kernel=kernel_size,
        channels=channels,
        dilation_rate=dilations[block*layers_per_block:
                                (block+1)*layers_per_block],
        activation=activation,
        dilation_channels=dilation_channels,
        residual=use_residual,
        skip_channels=skip_channels,
        l2_reg_factor=l2_reg_factor,
        dropout=dropout,
        condition=conditioning is not None)
      for block in range(blocks)]

    # create final layers
    self.final = [
      tf.keras.layers.Conv1D(
        filters=channel,
        kernel_size=1,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(l2_reg_factor))
      for channel in final_layers_channels]

    self.final.append(
      tf.keras.layers.Conv1D(
        filters=num_mixtures*3 if num_mixtures is not None else 2**bits,
        kernel_size=1,
        activation='softmax' if num_mixtures is None else 'linear',
        kernel_regularizer=tf.keras.regularizers.L2(l2_reg_factor))
    )

    # Compute receptive field
    self.receptive_field = 1+sum(dilations)*(kernel_size-1)+1

    # if conditioning is used, create conditioning layers
    if mapping_layers is None:
      mapping_layers = []
    elif isinstance(mapping_layers, int):
      mapping_layers = [mapping_layers]
    elif not isinstance(mapping_layers, list):
      raise ValueError('Mapping layers must be a list of integers.')
    if conditioning == 'local':
      self.mapping = tf.keras.Sequential(
        [tf.keras.layers.Lambda(tf.expand_dims, arguments={'axis':-1})]
      )
      self.mapping.add(layers=[
        tf.keras.layers.Conv1D(
          kernel=1, filters= layer,
          activation=mapping_activation,
          kernel_regularizer=tf.keras.regularizers.L2(l2_reg_factor),
        ) for layer in mapping_layers])
    elif conditioning == 'global':
      self.mapping = tf.keras.Sequential(layers=[
        tf.keras.layers.Dense(
          units=layer,
          activation=mapping_activation,
          kernel_regularizer=tf.keras.regularizers.L2(l2_reg_factor),
        ) for layer in mapping_layers])
      self.mapping.add(tf.keras.layers.Identity())
    self.conditioning = conditioning

    if num_mixtures is None:
      self.prepare_target = tf.keras.layers.Discretization(
        bin_boundaries=np.linspace(-1, 1, num=2**bits+1).tolist()[1:-1],
        num_bins=2**bits)
    else:
      self.prepare_target = lambda x: x

  def compile(self, **kwargs):
    """Compile function for the model
    """
    if 'loss' in kwargs:
      raise ValueError('Loss must be set in the model init function.')
    super(WaveNet, self).compile(**kwargs)

  def build(self, input_shape):
    """Build function for the model
    """
    def sequential_output_shape(model,shape):
      """Compute output shape for the model
      """
      for layer in model.layers:
        shape = layer.compute_output_shape(shape)
      return shape


    if self.conditioning == 'local':
      self.mapping.build(input_shape[1])
      cond_shape = sequential_output_shape(self.mapping,input_shape[1])
      x_shape = input_shape[0]
      cond_shape = (cond_shape[0],
                    cond_shape[1]*(x_shape[1]//cond_shape[1]),
                    cond_shape[2])
    elif self.conditioning == 'global':
      self.mapping.build(input_shape[1])
      cond_shape = sequential_output_shape(self.mapping,input_shape[1])
      x_shape = input_shape[0]
      cond_shape = (cond_shape[0], x_shape[1], cond_shape[1])
    else:
      x_shape = input_shape

    self.causal.build(x_shape)
    x_shape = self.causal.compute_output_shape(x_shape)

    for block in self.wavenet_blocks:
      if self.conditioning is not None:
        x_shape = [x_shape,cond_shape]
      block.build(x_shape)
      x_shape, skip_shape = block.compute_output_shape(x_shape)
    if self.use_skip:
      x_shape = skip_shape
    for layer in self.final:
      layer.build(x_shape)
      x_shape = layer.compute_output_shape(x_shape)
    self.optimizer.build(self.trainable_variables)

    self.built = True

  def call(self, inputs, training=False):
    """Call function for the model
    """
    if self.conditioning == 'local':
      condition = self.mapping(inputs[1])
      x = inputs[0]
      upsample = x.shape[1]//condition.shape[1]
      condition = tf.repeat(condition, upsample, axis=1)
    elif self.conditioning == 'global':
      condition = self.mapping(inputs[1])
      x = inputs[0]
      condition = tf.expand_dims(condition, axis=1)
      condition = tf.repeat(condition, x.shape[1], axis=1)
    else:
      x = inputs
    x = self.causal(x)
    skips = []
    for block in self.wavenet_blocks:
      if self.conditioning is not None:
        x = [x,condition]
      x, skip = block(x, training=training)
      skips.append(skip)
    if self.use_skip:
      x = tf.keras.layers.add(skips)
    for layer in self.final:
      x = layer(x)
    return x

  @tf.function
  def _generation(self, x, use_queues=False):
    """Generate one sample from model

    Args:
        x (tf.Tensor): input tensor or tuple
        use_queues (bool): use queues for generation
    """
    if use_queues:
      raise NotImplementedError('Generation not implemented yet.')

    predictions = self(x, training=False)
    pred = predictions[:,-1,:]
    pred = tf.expand_dims(pred, axis=1)
    sample = self.sample_waveform(pred)
    return sample

  def generate(self, length, batch_size: int = 1,
               condition = None, sample = None,
               use_queues=False):
    """Generate audio from model

    Args:
        length (int): length of the audio
        batch_size (int): batch size for generation 
          (ignored if sample or condition is provided)
        condition (tf.Tensor): condition tensor
        sample (tf.Tensor): sample tensor (instead of noise)
        use_queues (bool): use queues for generation
    """
    if self.conditioning is not None and condition is None:
      raise ValueError('Conditioning must be provided.')
    if condition is not None and sample is not None:
      if tf.shape(condition)[0] != tf.shape(sample)[0]:
        raise ValueError('Condition and sample must have same batch size.')
    if not use_queues:
      # prepare description
      desc = 'Generating '
      desc += 'from sample' if sample is not None else 'from noise'
      desc += ' with' if use_queues else ' without'
      desc += ' queues'

      if condition is not None:
        batch_size = tf.shape(condition)[0]
      if sample is None:
        sample = tf.random.stateless_normal((batch_size,self.receptive_field,1),
                                            seed=(4,2))
      x = sample
      output = []

      for _ in tqdm(range(length),
                    desc=desc,total=length,
                    unit='samples',unit_scale=True):
        if self.conditioning is not None:
          inputs = [x,condition]
        else:
          inputs = x
        predicted = self._generation(inputs, use_queues=False)
        output.append(predicted)
        x = tf.concat([x[:,1:],predicted],axis=1)
      return tf.concat(output,axis=1)
    raise NotImplementedError('Generation not implemented yet.')

  def train_step(self, data):
    """Train step for the model

    Args:
        data: input data, if conditioning is used, it should be a tuple
    """
    if self.conditioning is not None:
      x, condition = data
    else:
      x = data
    y_true = x[:, 1:, :]
    target = self.prepare_target(x[:, 1:, :])
    inputs = x[:, :-1, :]

    if self.conditioning is not None:
      inputs=[inputs,condition]

    with tf.GradientTape() as tape:
      pred = self(inputs, training=True)
      loss = tf.nn.compute_average_loss(
        self.loss_fn(target, pred))
      loss_final = loss
      if self.regularization:
        reg_loss = tf.nn.scale_regularization_loss(
          tf.reduce_sum(self.losses))
        loss_final += reg_loss
    gradients = tape.gradient(loss_final, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    sample = self.sample_waveform(pred)
    for metric in self.metrics:
      metric.update_state(y_true, sample)

    out_dict = {m.name: m.result() for m in self.metrics}
    out_dict['loss'] = loss
    if self.regularization:
      out_dict['regularization'] = reg_loss
    return out_dict

  def test_step(self, data):
    """Test step for the model
    
    Args:
        data: input data, if conditioning is used, it should be a tuple
    """
    if self.conditioning is not None:
      x, condition = data
    else:
      x = data
    y_true = x[:, 1:, :]
    target = self.prepare_target(x[:, 1:, :])
    inputs = x[:, :-1, :]

    if self.conditioning is not None:
      inputs=[inputs,condition]

    pred = self(inputs, training=False)
    loss = tf.nn.compute_average_loss(
        self.loss_fn(target, pred))
    sample = self.sample_waveform(pred)
    for metric in self.metrics:
      metric.update_state(y_true, sample)

    out_dict = {m.name: m.result() for m in self.metrics}
    out_dict['loss'] = loss
    return out_dict

  def sample_waveform(self, inputs):
    """Sample waveform from network output

    Args:
        inputs (tf.Tensor): prediction tensor
    Returns:
        tf.Tensor: sampled waveform, shape same as pred but last
          dimension is 1
    """
    if self.sampling_function == 'categorical':
      @tf.function(input_signature=[tf.TensorSpec(shape=(None,None),
                                                  dtype=tf.float32)])
      def sampling_fn(pred):
        samples = tf.random.stateless_categorical(tf.math.log(pred),1,
                                                  seed=(4,2),dtype=tf.int32)
        samples = tf.cast(samples,tf.float32)
        return samples/2.0**(self.bits-1) - 1.0
    elif self.sampling_function == 'gaussian':
      @tf.function(input_signature=[tf.TensorSpec(shape=(None,None),
                                                  dtype=tf.float32)])
      def sampling_fn(pred):
        weights, means, log_scales = tf.split(pred, 3, axis=-1)

        weights = tf.nn.softmax(weights, axis=-1)
        weights = tf.math.log(weights)
        selected = tf.random.stateless_categorical(weights,1,seed=(4,2),
                                                   dtype=tf.int32)
        selected = tf.squeeze(selected,axis=-1)


        selected = tf.one_hot(selected,depth=self.num_mixtures)
        mu = tf.reduce_sum(selected*means,axis=-1)
        scale = tf.reduce_sum(selected*tf.exp(log_scales),axis=-1)

        z = tf.random.stateless_normal(shape=tf.shape(mu),seed=(4,2))

        samples = mu + z *scale

        samples = tf.expand_dims(samples,axis=-1)

        return tf.clip_by_value(samples,-1,1)
    elif self.sampling_function == 'logistic':
      @tf.function(input_signature=[tf.TensorSpec(shape=(None,None),
                                                  dtype=tf.float32)])
      def sampling_fn(pred):
        weights, means, log_scales = tf.split(pred, 3, axis=-1)

        weights = tf.nn.softmax(weights, axis=-1)
        weights = tf.math.log(weights)
        selected = tf.random.stateless_categorical(weights,1,
                                                   seed=(4,2),dtype=tf.int32)
        selected = tf.squeeze(selected,axis=-1)


        selected = tf.one_hot(selected,depth=self.num_mixtures)
        mu = tf.reduce_sum(selected*means,axis=-1)
        scale = tf.reduce_sum(selected*tf.exp(log_scales),axis=-1)

        z = tf.random.stateless_uniform(shape=tf.shape(mu),seed=(4,2))

        samples = mu + scale*(tf.math.log(z)-tf.math.log(1-z))

        samples = tf.clip_by_value(samples,-1,1)
        samples = tf.expand_dims(samples,axis=-1)
        return samples
    else:
      raise NotImplementedError(f'Sampling {self.sampling_function}'+
        ' not implemented yet.')
    outputs = tf.map_fn(sampling_fn, inputs)
    return outputs #tf.transpose(out, [1, 2, 0])

  @tf.function
  def loss_fn(self,target, pred):
    """Loss function for the model

    Args:
        target (tf.Tensor): target tensor
        pred (tf.Tensor): prediction tensor
    Returns:
        tf.Tensor: loss tensor
    """
    if self.sampling_function == 'categorical':
      out = tf.keras.losses.sparse_categorical_crossentropy(target, pred)
    elif self.sampling_function == 'gaussian':
      weights, means, log_scales = tf.split(pred, 3, axis=-1)
      target = tf.repeat(target, self.num_mixtures, axis=-1)

      weights = tf.nn.softmax(weights, axis=-1)

      log_scales = tf.maximum(log_scales, -7) # to avoid NaNs - as in PixelCNN++
      scales = tf.exp(log_scales)

      x = tf.minimum((target-means)/scales,1e8)
      likelihood = tf.reduce_sum(
        weights*(
          tf.exp(-0.5*tf.square(x))/(scales*SQRT2PI)
        ),
        axis=-1)
      out = -1.0*tf.math.log(likelihood)
    elif self.sampling_function == 'logistic':
      weights, means, log_scales = tf.split(pred, 3, axis=-1)
      target = tf.repeat(target, self.num_mixtures, axis=-1)
      weights = tf.nn.softmax(weights, axis=-1)

      halfbit = 0.5*1/(2**self.bits) # as ints are converted to floats

      log_scales = tf.maximum(log_scales, -7) # to avoid NaNs - as in PixelCNN++
      likelihood = tf.reduce_sum(
        weights*(
          tf.nn.sigmoid((target-means+halfbit)*tf.exp(-1.0*log_scales))
          - tf.nn.sigmoid((target-means-halfbit)*tf.exp(-1.0*log_scales))
        ),
        axis=-1)
      out = -1.0*tf.math.log(likelihood)
    else:
      raise NotImplementedError(f'Loss {self.sampling_function}'+
        ' not implemented.')
    return out

  def compute_receptive_field(self,sampling_frequency):
    """Compute receptive field of the model
    """
    return self.receptive_field/sampling_frequency
    