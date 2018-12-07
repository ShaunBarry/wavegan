from __future__ import print_function
import _pickle as pickle
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

import loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator
from functools import reduce
import argparse
import glob

_FS = 16000
_WINDOW_LEN = 16384
_D_Z = 1024#100

from IPython.display import Audio
from IPython.core.display import display

parser = argparse.ArgumentParser()

parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
parser.add_argument('train_dir', type=str,
  help='Training directory')

data_args = parser.add_argument_group('Data')
data_args.add_argument('--data_dir', type=str,
  help='Data directory')
data_args.add_argument('--data_first_window', action='store_true', dest='data_first_window',
  help='If set, only use the first window from each audio example')

wavegan_args = parser.add_argument_group('WaveGAN')
wavegan_args.add_argument('--wavegan_kernel_len', type=int,
  help='Length of 1D filter kernels')
wavegan_args.add_argument('--wavegan_dim', type=int,
  help='Dimensionality multiplier for model of G and D')
wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
  help='Enable batchnorm')
wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
  help='Number of discriminator updates per generator update')
wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
  help='Which GAN loss to use')
wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn', 'lin', 'cub'],
  help='Generator upsample strategy')
wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
  help='If set, use post-processing filter')
wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,
  help='Length of post-processing filter for DCGAN')
wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,
  help='Radius of phase shuffle operation')

train_args = parser.add_argument_group('Train')
train_args.add_argument('--train_batch_size', type=int,
  help='Batch size')
train_args.add_argument('--train_save_secs', type=int,
  help='How often to save model')
train_args.add_argument('--train_summary_secs', type=int,
  help='How often to report summaries')

preview_args = parser.add_argument_group('Preview')
preview_args.add_argument('--preview_n', type=int,
  help='Number of samples to preview')

incept_args = parser.add_argument_group('Incept')
incept_args.add_argument('--incept_metagraph_fp', type=str,
  help='Inference model for inception score')
incept_args.add_argument('--incept_ckpt_fp', type=str,
  help='Checkpoint for inference model')
incept_args.add_argument('--incept_n', type=int,
  help='Number of generated examples to test')
incept_args.add_argument('--incept_k', type=int,
  help='Number of groups to test')

parser.set_defaults(
data_dir=None,
data_first_window=False,
wavegan_kernel_len=25,
wavegan_dim=64,
wavegan_batchnorm=False,
wavegan_disc_nupdates=5,
wavegan_loss='wgan-gp',
wavegan_genr_upsample='zeros',
wavegan_genr_pp=False,
wavegan_genr_pp_len=512,
wavegan_disc_phaseshuffle=2,
train_batch_size=64,
train_save_secs=300,
train_summary_secs=120,
preview_n=32,
incept_metagraph_fp='./eval/inception/infer.meta',
incept_ckpt_fp='./eval/inception/best_acc-103005',
incept_n=5000,
incept_k=10)

args = parser.parse_args(['train', './piano_training1rnn9', '--data_dir', 'piano', '--wavegan_genr_pp', '--wavegan_disc_phaseshuffle', '0'])
setattr(args, 'wavegan_g_kwargs', {
      'kernel_len': args.wavegan_kernel_len,
      'dim': args.wavegan_dim,
      'use_batchnorm': args.wavegan_batchnorm,
      'upsample': args.wavegan_genr_upsample
  })
setattr(args, 'wavegan_d_kwargs', {
      'kernel_len': args.wavegan_kernel_len,
      'dim': args.wavegan_dim,
      'use_batchnorm': args.wavegan_batchnorm,
      'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
  })

