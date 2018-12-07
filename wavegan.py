import tensorflow as tf
from tensorflow_cqt import *

def conv1d_transpose(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample='zeros'):
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_width),
        strides=(1, stride),
        padding='same'
        )[:, 0]
  elif upsample == 'nn':
    batch_size = tf.shape(inputs)[0]
    _, w, nch = inputs.get_shape().as_list()

    x = inputs

    x = tf.expand_dims(x, axis=1)
    x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
    x = x[:, 0]

    return tf.layers.conv1d(
        x,
        filters,
        kernel_width,
        1,
        padding='same')
  else:
    raise NotImplementedError

"""
  Input: [None, 100]
  Output: [None, 16384, 1]
"""
def WaveGANGenerator(
    z,
    gru_layer=tf.keras.layers.CuDNNGRU(1024, return_sequences=True),
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False,
    length=16,
    return_gru=False,
    use_sequence=False,
    reuse=False):
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  output = z
  with tf.variable_scope('z_project', reuse=reuse):
    #output = tf.layers.dense(output, 4 * 4 * dim * 16)
    
    if not use_sequence:
        output = tf.keras.layers.RepeatVector(length)(output)
        print('output.shape:', output.get_shape().as_list())
    #output = tf.reshape(output, [batch_size, 16*(dim//64), 64 * 16])
    output = gru_layer(output, initial_state=None)
    print('output.shape:', output.get_shape().as_list())
    output = batchnorm(output)
  #output = tf.nn.relu(output)
  #output = tf.contrib.layers.conv1d(output,1024,1)
  output1 = tf.layers.Conv1D(1024,1,activation=tf.nn.relu)(output)
  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0', reuse=reuse):
    output = conv1d_transpose(output1, dim * 8, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)
  print('output.shape:', output.get_shape().as_list())

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1', reuse=reuse):
    output = conv1d_transpose(output, dim * 4, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)
  print('output.shape:', output.get_shape().as_list())

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2', reuse=reuse):
    output = conv1d_transpose(output, dim * 2, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)
  print('output.shape:', output.get_shape().as_list())

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3', reuse=reuse):
    output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)
  print('output.shape:', output.get_shape().as_list())

  # Layer 4
  # [4096, 64] -> [16384, 1]
  with tf.variable_scope('upconv_4', reuse=reuse):
    output = conv1d_transpose(output, 1, kernel_len, 4, upsample=upsample)
  output = tf.nn.tanh(output)
  print('output.shape:', output.get_shape().as_list())

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if len(update_ops) != 10:
      raise Exception('Other update ops found in graph')
    with tf.control_dependencies(update_ops):
      output = tf.identity(output)
  if return_gru:
    return output, output1
  else:
    return output


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    disc_reps=['time'],
    x_cqt=None):#,'cqt']):
  
  batch_size = tf.shape(x)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x
        
  if 'time' in disc_reps:


      # Layer 0
      # [16384, 1] -> [4096, 64]
      output = x
      with tf.variable_scope('downconv_0'):
        output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
      output = lrelu(output)
      output = phaseshuffle(output)

      # Layer 1
      # [4096, 64] -> [1024, 128]
      with tf.variable_scope('downconv_1'):
        output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)

      # Layer 2
      # [1024, 128] -> [256, 256]
      with tf.variable_scope('downconv_2'):
        output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)

      # Layer 3
      # [256, 256] -> [64, 512]
      with tf.variable_scope('downconv_3'):
        output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)

      # Layer 4
      # [64, 512] -> [16, 1024]
      with tf.variable_scope('downconv_4'):
        output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
      output = lrelu(output)

      # Flatten
      #output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])
      output_time = tf.contrib.layers.flatten(output)
      print('output_time.shape:',output_time.shape)
  if 'cqt' in disc_reps:
    
      SR = 16000
      HOP_LEN = 256
      FMIN = 32.7
      N_BINS = 360
      BINS_PER_OCTAVE = 60
        
      if x_cqt == None:
        x_cqt = tf.log1p(tf.abs(tf.contrib.signal.stft(x[:,:,0],512,128,fft_length=512)))[:,:,:,tf.newaxis]#tf.reshape(tf.abs(tf.fft(tf.cast(x[:,:,0],tf.complex64))), [1, 256,64,1])#
      #tf.stop_gradient(x_cqt)
#tf.reshape(x,[batch_size,256,64,1])#tf.log1p(tf.abs(cqt_tf(x[:,:,:,tf.newaxis], sr=SR, hop_length=HOP_LEN, fmin=FMIN, n_bins=N_BINS,
            #bins_per_octave=BINS_PER_OCTAVE, scale=True, use_smoothing=True)))[:,:,:,tf.newaxis]
    
      print('x_cqt.shape:',x_cqt.shape)
      kernel = [5,11]
      output = x_cqt
      with tf.variable_scope('downconv_0_cqt'):
        output = tf.layers.conv2d(output, dim, kernel, [2,2], padding='SAME')
      output = lrelu(output)
      output = phaseshuffle(output)
      print('cqt_output.shape:',output.shape)

      # Layer 1
      # [4096, 64] -> [1024, 128]
      with tf.variable_scope('downconv_1_cqt'):
        output = tf.layers.conv2d(output, dim * 2, kernel, [2,2], padding='SAME')
        output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)
      print('cqt_output.shape:',output.shape)

      # Layer 2
      # [1024, 128] -> [256, 256]
      with tf.variable_scope('downconv_2_cqt'):
        output = tf.layers.conv2d(output, dim * 4, kernel, [2,2], padding='SAME')
        output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)
      print('cqt_output.shape:',output.shape)

      # Layer 3
      # [256, 256] -> [64, 512]
      with tf.variable_scope('downconv_3_cqt'):
        output = tf.layers.conv2d(output, dim * 8, kernel, [2,2], padding='SAME')
        output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)
      print('cqt_output.shape:',output.shape)

      # Layer 4
      # [64, 512] -> [16, 1024]
      with tf.variable_scope('downconv_4_cqt'):
        output = tf.layers.conv2d(output, dim * 16, kernel, [2,2], padding='SAME')
        output = batchnorm(output)
      output = lrelu(output)
      print('cqt_output.shape:',output.shape)

      # Flatten
      #output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])
      output_cqt = tf.contrib.layers.flatten(output)
      print('output_cqt.shape:',output_cqt.shape)
    
  if disc_reps == ['time','cqt']:
    output = tf.concat([output_time, output_cqt], axis=-1)
    print('total_output.shape:',output.shape)
  elif disc_reps == ['cqt']:
    output = output_cqt
  else:
    output = output_time
    
  # Connect to single logit
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
