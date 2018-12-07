from __future__ import division

import warnings
import numpy as np
import scipy.fftpack as fft
from numba import jit

from librosa.core import audio
from librosa.core.time_frequency import cqt_frequencies, note_to_hz
from librosa.core.spectrum import stft
from librosa.core.pitch import estimate_tuning
from librosa import cache
from librosa import filters
from librosa import util
from librosa.util.exceptions import ParameterError
import librosa

import tensorflow as tf

from scipy.signal import general_gaussian


def bicubic_interp_2d(input_, new_size, endpoint=False):
    """
    Args :
    input_ : Input tensor. Its shape should be
        [batch_size, height, width, channel].
        In this implementation, the shape should be fixed for speed.
    new_size : The output size [new_height, new_width]
    ref : 
    http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
    """

    shape = input_.get_shape().as_list()
    batch_size = shape[0]
    height  = shape[1]
    width   = shape[2]
    channel = shape[3]

    def _hermite(A, B, C, D, t):
        a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
        b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
        c = A * (-0.5) + C * 0.5
        d = B

        return a*t*t*t + b*t*t + c*t + d

    def _get_grid_array(n_i, y_i, x_i, c_i):
        n, y, x, c = np.meshgrid(n_i, y_i, x_i, c_i, indexing='ij')
        n = np.expand_dims(n, axis=4)
        y = np.expand_dims(y, axis=4)
        x = np.expand_dims(x, axis=4)
        c = np.expand_dims(c, axis=4)

        return np.concatenate([n,y,x,c], axis=4)

    def _get_frac_array(y_d, x_d, n, c):
        y = y_d.shape[0]
        x = x_d.shape[0]
        y_t = y_d.reshape([1, -1, 1, 1])
        x_t = x_d.reshape([1, 1, -1, 1])
        y_t = tf.constant(np.tile(y_t, (n,1,x,c)), dtype=tf.float32)
        x_t = tf.constant(np.tile(x_t, (n,y,1,c)), dtype=tf.float32)
        return y_t, x_t

    def _get_index_tensor(grid, x, y):
        new_grid = np.array(grid)

        grid_y = grid[:,:,:,:,1] + y
        grid_x = grid[:,:,:,:,2] + x

        grid_y = np.clip(grid_y, 0, height-1)
        grid_x = np.clip(grid_x, 0, width-1)

        new_grid[:,:,:,:,1] = grid_y
        new_grid[:,:,:,:,2] = grid_x

        return tf.constant(new_grid, dtype=tf.int32)

    new_height = new_size[0]
    new_width  = new_size[1]

    n_i = np.arange(batch_size)
    c_i = np.arange(channel)

    if endpoint:
        y_f = np.linspace(0., height-1, new_height)
    else:
        y_f = np.linspace(0., height, new_height, endpoint=False)
    y_i = y_f.astype(np.int32)
    y_d = y_f - np.floor(y_f)

    if endpoint:
        x_f = np.linspace(0., width-1, new_width)
    else:
        x_f = np.linspace(0., width, new_width, endpoint=False)
    x_i = x_f.astype(np.int32)
    x_d = x_f - np.floor(x_f) 

    grid = _get_grid_array(n_i, y_i, x_i, c_i)
    y_t, x_t = _get_frac_array(y_d, x_d, batch_size, channel)

    i_00 = _get_index_tensor(grid, -1, -1)
    i_10 = _get_index_tensor(grid, +0, -1)
    i_20 = _get_index_tensor(grid, +1, -1)
    i_30 = _get_index_tensor(grid, +2, -1)

    i_01 = _get_index_tensor(grid, -1, +0)
    i_11 = _get_index_tensor(grid, +0, +0)
    i_21 = _get_index_tensor(grid, +1, +0)
    i_31 = _get_index_tensor(grid, +2, +0)

    i_02 = _get_index_tensor(grid, -1, +1)
    i_12 = _get_index_tensor(grid, +0, +1)
    i_22 = _get_index_tensor(grid, +1, +1)
    i_32 = _get_index_tensor(grid, +2, +1)

    i_03 = _get_index_tensor(grid, -1, +2)
    i_13 = _get_index_tensor(grid, +0, +2)
    i_23 = _get_index_tensor(grid, +1, +2)
    i_33 = _get_index_tensor(grid, +2, +2)

    p_00 = tf.gather_nd(input_, i_00)
    p_10 = tf.gather_nd(input_, i_10)
    p_20 = tf.gather_nd(input_, i_20)
    p_30 = tf.gather_nd(input_, i_30)

    p_01 = tf.gather_nd(input_, i_01)
    p_11 = tf.gather_nd(input_, i_11)
    p_21 = tf.gather_nd(input_, i_21)
    p_31 = tf.gather_nd(input_, i_31)

    p_02 = tf.gather_nd(input_, i_02)
    p_12 = tf.gather_nd(input_, i_12)
    p_22 = tf.gather_nd(input_, i_22)
    p_32 = tf.gather_nd(input_, i_32)

    p_03 = tf.gather_nd(input_, i_03)
    p_13 = tf.gather_nd(input_, i_13)
    p_23 = tf.gather_nd(input_, i_23)
    p_33 = tf.gather_nd(input_, i_33)

    col0 = _hermite(p_00, p_10, p_20, p_30, x_t)
    col1 = _hermite(p_01, p_11, p_21, p_31, x_t)
    col2 = _hermite(p_02, p_12, p_22, p_32, x_t)
    col3 = _hermite(p_03, p_13, p_23, p_33, x_t)
    value = _hermite(col0, col1, col2, col3, y_t)

    return value


def audio_resample_tf(y, orig_sr, target_sr, res_type='kaiser_best', fix=True, scale=False, use_smoothing=True, use_bicubic=False):
    ratio = float(target_sr) / orig_sr
    y_length = y.get_shape().as_list()[1]
    n_samples = int(np.ceil(y_length * ratio))
    
    if use_smoothing:
        window = general_gaussian(5, 0.5, True).astype(np.float32)
        window /= window.sum()

        window_tf = tf.constant(window[:, np.newaxis, np.newaxis, np.newaxis], dtype=tf.float32)
        y = tf.nn.conv2d(y,
                     window_tf,
                     strides=[1,1,1,1],
                     padding="SAME")
    
    if (not use_bicubic) and (np.mod(np.log2(float(orig_sr)/target_sr), 1) == 0.0):
        for i in range(int(np.log2(float(orig_sr)/target_sr)) - 1):
            y = y[:, ::2, :, :]
            y = tf.nn.conv2d(y,
                     window_tf,
                     strides=[1,1,1,1],
                     padding="SAME")
        y_hat = y[:, ::2, :, :]
    else:
        y_hat = bicubic_interp_2d(y, [n_samples, 1], endpoint=False)
        
    
    if scale:
        y_hat /= np.sqrt(ratio)
    return y_hat

def __early_downsample_tf(y, sr, hop_length, res_type, n_octaves,
                       nyquist, filter_cutoff, scale, use_smoothing):
    '''Perform early downsampling on an audio signal, if it applies.'''

    downsample_count = __early_downsample_count(nyquist, filter_cutoff,
                                                hop_length, n_octaves)

    
    if downsample_count > 0 and res_type == 'kaiser_fast':
        downsample_factor = 2**(downsample_count)

        hop_length //= downsample_factor

        if y.get_shape().as_list()[1] < downsample_factor:
            raise ParameterError('Input signal length={:d} is too short for '
                                 '{:d}-octave CQT'.format(len(y), n_octaves))

        new_sr = sr / float(downsample_factor)
        
        print('Early downsample, from sr:', sr, 'to new_sr:', new_sr)
        y = audio_resample_tf(y, sr, new_sr, scale=scale, use_smoothing=use_smoothing)

        # If we're not going to length-scale after CQT, we
        # need to compensate for the downsampling factor here
        if not scale:
            y *= np.sqrt(downsample_factor)

        sr = new_sr

    return y, sr, hop_length

def __cqt_response_tf(y, n_fft, hop_length, fft_basis_tf, mode, debug=False):
    '''Compute the filter response with a target STFT hop.'''

    hop_len = hop_length
    n_samples = y.get_shape().as_list()[1]
    num_bins = np.ceil(float(n_samples)/hop_len)
    n_samples_new = int(num_bins*hop_len)
    n_samples_new = int(n_samples_new + hop_len*(num_bins - int(np.ceil((n_samples_new-n_fft+1)/hop_len))))
    
    additional_samples = n_samples_new - n_samples
    if additional_samples % 2 == 0:
        paddings = tf.constant([[0,0],
                                [additional_samples//2, additional_samples//2]])
    else:
        paddings = tf.constant([[0,0],
                                [additional_samples//2 + 1, additional_samples//2]])
    y = tf.pad(y[:,:,0,0], paddings, "CONSTANT")
    
    # Compute the STFT matrix
    D = tf.contrib.signal.stft(y, n_fft, hop_length, n_fft, window_fn=tf.ones, pad_end=False)
    if debug:
        print('Getting response, n_fft:', n_fft, 
              'hop_len:', hop_len, 
              'n_samples:', n_samples, 
              'y_shape', y.get_shape().as_list(),
              'D_shape', D.get_shape().as_list())

    
    #fft_basis_tf = tf.transpose(denseNDArrayToSparseTensor(fft_basis), [1,0])
    # And filter response energy
    d_proj = tf.tensordot(D, fft_basis_tf, [[len(D.get_shape().as_list()) - 1], [0]])
    return tf.transpose(d_proj, [0, 2, 1]), additional_samples

def __icqt_response_tf(d_proj, n_fft, hop_length, fft_basis_tf, mode, samples_added):
    '''Compute the filter response with a target STFT hop.'''

    d_proj = tf.transpose(d_proj, [0, 2, 1])
    D = tf.tensordot(d_proj, fft_basis_tf, [[len(d_proj.get_shape().as_list()) - 1], [0]])
    
    # Compute the STFT matrix

    y = tf.contrib.signal.inverse_stft(D, n_fft, hop_length, n_fft, window_fn=None)
    
    if samples_added % 2 == 0:
        y = y[:,samples_added//2:-samples_added//2]
    else:
        y = y[:,samples_added//2-1:-samples_added//2]

    return y

def __icqt_response(d_proj, n_fft, hop_length, fft_basis, mode):
    '''Compute the filter response with a target STFT hop.'''

    D = fft_basis.dot(d_proj)
    # Compute the STFT matrix

    y = librosa.core.istft(D, hop_length=hop_length, win_length=n_fft, window='ones')

    return y

def __trim_stack_tf(cqt_resp, n_bins):
    '''Helper function to trim and stack a collection of CQT responses'''

    # cleanup any framing errors at the boundaries
    my_list = [x.get_shape().as_list() for x in cqt_resp]

    max_col = min(x.get_shape().as_list()[2] for x in cqt_resp)

    cqt_resp = tf.concat([x[:, :, :max_col] for x in cqt_resp][::-1], axis=1)

    # Finally, clip out any bottom frequencies that we don't really want
    # Transpose magic here to ensure column-contiguity
    return cqt_resp[-n_bins:]

def cqt_tf(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
        bins_per_octave=12, filter_scale=1,
        norm=1, sparsity=0.01, window='hann',
        scale=True,
        pad_mode='reflect', use_smoothing=True, return_added_samples=False, debug=False):

    tuning = 0.0
    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    len_orig = y.get_shape().as_list()[1]

    added_samples = []
    
    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)
    filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)
    nyquist = sr / 2.0
    if filter_cutoff < audio.BW_FASTEST * nyquist:
        res_type = 'kaiser_fast'
    else:
        res_type = 'kaiser_best'

    y, sr, hop_length = __early_downsample_tf(y, sr, hop_length,
                                           res_type,
                                           n_octaves,
                                           nyquist, filter_cutoff, scale, use_smoothing)

    #print('y after early downsaple:', y.get_shape().as_list()[1])
    cqt_resp = []

    if res_type != 'kaiser_fast':

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                               n_filters,
                                               bins_per_octave,
                                               tuning,
                                               filter_scale,
                                               norm,
                                               sparsity,
                                               window=window)

        fft_basis = fft_basis.astype('complex64')
    
        fft_basis_tf = tf.constant(fft_basis, dtype=tf.complex64)
        fft_basis_tf = tf.transpose(fft_basis_tf)
        # Compute the CQT filter response and append it to the stack
        cqt_res, add_samples = __cqt_response_tf(y, n_fft, hop_length, fft_basis_tf, pad_mode, debug)
        cqt_resp.append(cqt_res)
        added_samples += [add_samples]

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)

        res_type = 'kaiser_fast'

    # Make sure our hop is long enough to support the bottom octave
    num_twos = __num_two_factors(hop_length)
    if num_twos < n_octaves - 1:
        raise ParameterError('hop_length must be a positive integer '
                             'multiple of 2^{0:d} for {1:d}-octave CQT'
                             .format(n_octaves - 1, n_octaves))

    # Now do the recursive bit
    fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                           n_filters,
                                           bins_per_octave,
                                           tuning,
                                           filter_scale,
                                           norm,
                                           sparsity,
                                           window=window)
    
    fft_basis = fft_basis.astype('complex64')
    
    fft_basis_tf = tf.constant(fft_basis, dtype=tf.complex64)
    fft_basis_tf = tf.transpose(fft_basis_tf)

    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):

        # Resample (except first time)
        if i > 0:
            if my_y.get_shape().as_list()[1] < 2:
                raise ParameterError('Input signal length={} is too short for '
                                     '{:d}-octave CQT'.format(len_orig,
                                                              n_octaves))

            #print('Resample from ', my_sr, 'to', my_sr/2.0)
            my_y = audio_resample_tf(my_y, my_sr, my_sr/2.0,
                                  res_type=res_type,
                                  scale=True, use_smoothing=use_smoothing)
            # The re-scale the filters to compensate for downsampling
            fft_basis_tf *= np.sqrt(2)

            my_sr /= 2.0
            my_hop //= 2
            #print('y after early downsaple:', my_y.get_shape().as_list()[1])

        # Compute the cqt filter response and append to the stack
        cqt_res, add_samples = __cqt_response_tf(my_y, n_fft, my_hop, fft_basis_tf, pad_mode, debug)
        cqt_resp.append(cqt_res)
        added_samples += [add_samples]

    C = __trim_stack_tf(cqt_resp, n_bins)

    if scale:
        lengths = filters.constant_q_lengths(sr, fmin,
                                             n_bins=n_bins,
                                             bins_per_octave=bins_per_octave,
                                             tuning=tuning,
                                             window=window,
                                             filter_scale=filter_scale)
        lengths_tf = tf.constant(lengths.astype('complex64'), dtype=tf.complex64)
        C /= tf.sqrt(lengths_tf[:, tf.newaxis])

    if return_added_samples:
        return C, added_samples
    else:
        return C

def icqt(C, sr=22050, hop_length=512, fmin=None, n_bins=84,
        bins_per_octave=12, filter_scale=1,
        norm=1, sparsity=0.01, window='hann',
        scale=True,
        pad_mode='reflect', use_smoothing=True):

    tuning = 0.0
    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    if scale:
        lengths = filters.constant_q_lengths(sr, fmin,
                                             n_bins=n_bins,
                                             bins_per_octave=bins_per_octave,
                                             tuning=tuning,
                                             window=window,
                                             filter_scale=filter_scale)
        C *= np.sqrt(lengths[:, np.newaxis])
    
    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)
    filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)
    nyquist = sr / 2.0
    if filter_cutoff < audio.BW_FASTEST * nyquist:
        res_type = 'kaiser_fast'
    else:
        res_type = 'kaiser_best'

    y = np.zeros((1000,))
    y, sr, hop_length = __early_downsample(y, sr, hop_length,
                                           res_type,
                                           n_octaves,
                                           nyquist, filter_cutoff, scale)

    cqt_resp = []

    for i in range(n_octaves):
        cqt_resp += [C[i*bins_per_octave:i*bins_per_octave+bins_per_octave,:]]
    
    cqt_resp = cqt_resp[::-1]
    
    if res_type != 'kaiser_fast':

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                               n_filters,
                                               bins_per_octave,
                                               tuning,
                                               filter_scale,
                                               norm,
                                               sparsity,
                                               window=window)

        fft_basis = np.linalg.pinv(fft_basis)
        # Compute the CQT filter response and append it to the stack
        y = __icqt_response(cqt_resp[0], n_fft, hop_length, fft_basis, pad_mode)

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)

        res_type = 'kaiser_fast'

    # Make sure our hop is long enough to support the bottom octave
    num_twos = __num_two_factors(hop_length)
    if num_twos < n_octaves - 1:
        raise ParameterError('hop_length must be a positive integer '
                             'multiple of 2^{0:d} for {1:d}-octave CQT'
                             .format(n_octaves - 1, n_octaves))

    # Now do the recursive bit
    fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                           n_filters,
                                           bins_per_octave,
                                           tuning,
                                           filter_scale,
                                           norm,
                                           sparsity,
                                           window=window)

    fft_basis = np.linalg.pinv(fft_basis)
    
    my_y, my_sr, my_hop = y, sr, hop_length

    y = 0.0
    # Iterate down the octaves
    for i in range(n_octaves):

        
        # Resample (except first time)
        if i > 0:
            
            #my_y = audio_resample_tf(my_y, my_sr, my_sr/2.0,
            #                      res_type=res_type,
            #                      scale=True, use_smoothing=use_smoothing)
            
            # The re-scale the filters to compensate for downsampling
            fft_basis /= np.sqrt(2)

            my_sr /= 2.0
            my_hop //= 2
            
        # Compute the cqt filter response and append to the stack
            my_y = __icqt_response(cqt_resp[i], n_fft, my_hop, fft_basis, pad_mode)
            my_y = audio.resample(my_y, my_sr, sr,
                               res_type=res_type,
                               scale=True)
            y += my_y
            
        else:
            my_y = __icqt_response(cqt_resp[i], n_fft, my_hop, fft_basis, pad_mode)
            y += my_y

        print('Octave:',i)
        print('y.size:', my_y.size)
        print('SR:', my_sr)
        print('Hop:', my_hop)
        print('New SR:',sr)
    return y

def icqt_tf(C, y, added_samples, sr=22050, hop_length=512, fmin=None, n_bins=84,
        bins_per_octave=12, filter_scale=1,
        norm=1, sparsity=0.01, window='hann',
        scale=True,
        pad_mode='reflect', use_smoothing=True, n_samples_total=None):

    tuning = 0.0
    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    if scale:
        lengths = filters.constant_q_lengths(sr, fmin,
                                             n_bins=n_bins,
                                             bins_per_octave=bins_per_octave,
                                             tuning=tuning,
                                             window=window,
                                             filter_scale=filter_scale)
        lengths_tf = tf.constant(lengths.astype('complex64'), dtype=tf.complex64)
        C *= tf.sqrt(lengths_tf[:, tf.newaxis])
    
    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)
    filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)
    nyquist = sr / 2.0
    if filter_cutoff < audio.BW_FASTEST * nyquist:
        res_type = 'kaiser_fast'
    else:
        res_type = 'kaiser_best'

    y, sr, hop_length = __early_downsample(y, sr, hop_length,
                                           res_type,
                                           n_octaves,
                                           nyquist, filter_cutoff, scale)

    cqt_resp = []

    for i in range(n_octaves):
        cqt_resp += [C[:,i*bins_per_octave:i*bins_per_octave+bins_per_octave,:]]
    
    cqt_resp = cqt_resp[::-1]
    
    if n_samples_total == None:
        n_bins = cqt_resp[0].get_shape().as_list()[-1]
        n_samples_total = hop_length*n_bins
    print('n_samples_total:', n_samples_total)
    
    if res_type != 'kaiser_fast':

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                               n_filters,
                                               bins_per_octave,
                                               tuning,
                                               filter_scale,
                                               norm,
                                               sparsity,
                                               window=window)

        fft_basis = np.linalg.pinv(fft_basis)
        fft_basis_tf = tf.transpose(tf.constant(fft_basis.astype(np.complex64)))
        # Compute the CQT filter response and append it to the stack
        y = __icqt_response_tf(cqt_resp[0], n_fft, hop_length, fft_basis_tf, pad_mode, added_samples[0])

        y = tf.image.resize_images(y[:,:,tf.newaxis, tf.newaxis], [n_samples_total,1])[:,:,0,0]
        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)

        res_type = 'kaiser_fast'

    # Make sure our hop is long enough to support the bottom octave
    num_twos = __num_two_factors(hop_length)
    if num_twos < n_octaves - 1:
        raise ParameterError('hop_length must be a positive integer '
                             'multiple of 2^{0:d} for {1:d}-octave CQT'
                             .format(n_octaves - 1, n_octaves))

    # Now do the recursive bit
    fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                           n_filters,
                                           bins_per_octave,
                                           tuning,
                                           filter_scale,
                                           norm,
                                           sparsity,
                                           window=window)

    fft_basis_tf = tf.transpose(tf.constant(np.linalg.pinv(fft_basis.astype(np.complex64))))
    
    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):

        
        # Resample (except first time)
        if i > 0:
            
            #my_y = audio_resample_tf(my_y, my_sr, my_sr/2.0,
            #                      res_type=res_type,
            #                      scale=True, use_smoothing=use_smoothing)
            
            # The re-scale the filters to compensate for downsampling
            my_sr /= 2.0
            my_hop //= 2
            
            ratio = float(sr)/my_sr
            
            
        # Compute the cqt filter response and append to the stack
            my_y = __icqt_response_tf(cqt_resp[i+1], n_fft, my_hop, fft_basis_tf/np.sqrt(ratio), pad_mode, added_samples[i+1])
            my_y = tf.image.resize_images(my_y[:,:,tf.newaxis, tf.newaxis], [n_samples_total,1])[:,:,0,0]/np.sqrt(ratio)
            
            y += my_y
            

            
        else:
            my_y = __icqt_response_tf(cqt_resp[i+1], n_fft, my_hop, fft_basis_tf, pad_mode, added_samples[i+1])
            my_y = tf.image.resize_images(my_y[:,:,tf.newaxis, tf.newaxis], [n_samples_total,1])[:,:,0,0]
            y += my_y

        #print('Octave:',i)
        #print('y.size:', my_y.get_shape().as_list())
        #print('SR:', my_sr)
        #print('Hop:', my_hop)
        #print('New SR:',sr)
    return y

def hcqt_tf(y, n_octaves=6, bins_per_octave=60, num_harmonics=5, fmin=32.7, sr=22050, hop_length=256):

    # How many bins do we need?
    n_bins_plane = N_OCTAVES * BINS_PER_OCTAVE

    n_bins_master = int(np.ceil(np.log2(np.max(HARMONICS))) * BINS_PER_OCTAVE) + n_bins_plane

    cqt_master = cqt_tf(y, n_bins=n_bins_master, fmin=fmin, hop_length=hop_length, 
                        sr=sr, bins_per_octave=bins_per_octave)

    inds124 = [np.where(master_cqt_freqs == np.array(h)*FMIN)[0][0] for h in [1,2,4]]

    hcqt_list = [cqt_master[:,inds124[0]:inds124[0]+n_bins_plane,:], 
                 cqt_master[:,inds124[1]:inds124[1]+n_bins_plane,:], 
                 0,
                 cqt_master[:,inds124[2]:inds124[2]+n_bins_plane,:], 
                 0]
    
    hcqt_list[2] = cqt_tf(y, n_bins=n_bins_master, fmin=fmin*3.0, hop_length=hop_length, 
                        sr=sr, bins_per_octave=bins_per_octave)
    
    hcqt_list[4] = cqt_tf(y, n_bins=n_bins_master, fmin=fmin*5.0, hop_length=hop_length, 
                        sr=sr, bins_per_octave=bins_per_octave)
    
    hcqt = tf.stack(hcqt_list, axis=-1)
    
    return hcqt

def __cqt_filter_fft(sr, fmin, n_bins, bins_per_octave, tuning,
                     filter_scale, norm, sparsity, hop_length=None,
                     window='hann'):
    '''Generate the frequency domain constant-Q filter basis.'''

    basis, lengths = filters.constant_q(sr,
                                        fmin=fmin,
                                        n_bins=n_bins,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning,
                                        filter_scale=filter_scale,
                                        norm=norm,
                                        pad_fft=True,
                                        window=window)

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if (hop_length is not None and
            n_fft < 2.0**(1 + np.ceil(np.log2(hop_length)))):

        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft // 2)+1]

    # sparsify the basis
    #fft_basis = util.sparsify_rows(fft_basis, quantile=sparsity)

    return fft_basis, n_fft, lengths


def __trim_stack(cqt_resp, n_bins):
    '''Helper function to trim and stack a collection of CQT responses'''

    # cleanup any framing errors at the boundaries
    max_col = min(x.shape[1] for x in cqt_resp)

    cqt_resp = np.vstack([x[:, :max_col] for x in cqt_resp][::-1])

    # Finally, clip out any bottom frequencies that we don't really want
    # Transpose magic here to ensure column-contiguity
    return np.ascontiguousarray(cqt_resp[-n_bins:].T).T


def __cqt_response(y, n_fft, hop_length, fft_basis, mode):
    '''Compute the filter response with a target STFT hop.'''

    # Compute the STFT matrix
    D = stft(y, n_fft=n_fft, hop_length=hop_length,
             window='ones',
             pad_mode=mode)

    # And filter response energy
    return fft_basis.dot(D)


def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    '''Compute the number of early downsampling operations'''

    downsample_count1 = max(0, int(np.ceil(np.log2(audio.BW_FASTEST * nyquist /
                                                   filter_cutoff)) - 1) - 1)

    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def __early_downsample(y, sr, hop_length, res_type, n_octaves,
                       nyquist, filter_cutoff, scale):
    '''Perform early downsampling on an audio signal, if it applies.'''

    downsample_count = __early_downsample_count(nyquist, filter_cutoff,
                                                hop_length, n_octaves)

    if downsample_count > 0 and res_type == 'kaiser_fast':
        downsample_factor = 2**(downsample_count)

        hop_length //= downsample_factor

        if len(y) < downsample_factor:
            raise ParameterError('Input signal length={:d} is too short for '
                                 '{:d}-octave CQT'.format(len(y), n_octaves))

        new_sr = sr / float(downsample_factor)
        y = audio.resample(y, sr, new_sr,
                           res_type=res_type,
                           scale=True)

        # If we're not going to length-scale after CQT, we
        # need to compensate for the downsampling factor here
        if not scale:
            y *= np.sqrt(downsample_factor)

        sr = new_sr

    return y, sr, hop_length


def __num_two_factors(x):
    """Return how many times integer x can be evenly divided by 2.

    Returns 0 for non-positive integers.
    """
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos


@jit(nopython=True)
def __activation_fill(x, basis, activation, hop_length):
    '''Helper function for icqt time-domain reconstruction'''

    n = len(x)
    n_fft = len(basis)
    n_frames = len(activation)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += activation[i] * basis[:max(0, min(n_fft, n - sample))]