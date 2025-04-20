# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 888

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'siMLPe'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]


C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))


exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'lib'))

"""Data Dir and Weight Dir"""
# TODO

"""Dataset Config"""
C.amass_anno_dir = osp.join(C.root_dir, 'data/amass/')
C.pw3d_anno_dir = osp.join(C.root_dir, 'data/3dpw/sequenceFiles/')
C.motion = edict()

C.motion.amass_input_length = 10
C.motion.amass_input_length_dct = C.motion.amass_input_length
C.motion.amass_target_length_train = 10
C.motion.amass_target_length_eval = 25
C.motion.dim = 54

C.motion.pw3d_input_length = 10
C.motion.pw3d_target_length_train = 10
C.motion.pw3d_target_length_eval = 25

C.data_aug = True
C.residual_output = True
C.use_relative_loss = True

""" Model Config"""
C.model = 'siMLPe_RNN'
## Network
C.pre_dct = True
C.post_dct = True
## Motion Network mlp
dim_ = C.motion.dim
C.motion_mlp = edict()
C.motion_mlp.hidden_dim = dim_
C.motion_mlp.seq_len = C.motion.amass_input_length_dct
C.motion_mlp.num_layers = 48
C.motion_mlp.with_normalization = True
C.motion_mlp.spatial_fc_only = False
C.motion_mlp.norm_axis = 'spatial'
## Motion Network FC In
C.motion_fc_in = edict()
C.motion_fc_in.in_features = C.motion.dim
C.motion_fc_in.out_features = dim_
C.motion_fc_in.with_norm = False
C.motion_fc_in.activation = 'relu'
C.motion_fc_in.init_w_trunc_normal = False
C.motion_fc_in.temporal_fc = False
## Motion Network FC Out
C.motion_fc_out = edict()
C.motion_fc_out.in_features = dim_
C.motion_fc_out.out_features = C.motion.dim
C.motion_fc_out.with_norm = False
C.motion_fc_out.activation = 'relu'
C.motion_fc_out.init_w_trunc_normal = True
C.motion_fc_out.temporal_fc = False

"""RNN Config"""
C.motion_rnn = edict()
C.motion_rnn.local_spatial_fc = True
C.motion_rnn.recursive_residual = True
C.motion_rnn.rnn_layers = 1
C.motion_rnn.rnn_blocks = 1
# C.motion_rnn.rnn_state_size = config.motion.dim
C.motion_rnn.rnn_state_size = int(config.motion.dim/2*3) # 81
C.motion_rnn.num_temp_blocks = 1 # must be larger than 1
# deprecated
# C.motion_rnn.with_normalization = False
C.motion_rnn.use_gru = True
C.motion_rnn.history_window_size = C.motion.amass_input_length
C.motion_rnn.encode_history = True
# must be larger than 1, smaller or equal to C.motion.amass_input_length
# C.motion_rnn.short_term_window_size = C.motion.amass_input_length
C.motion_rnn.short_term_window_size = 10
# deprecated
# C.motion_rnn.sliding_long_term = False
C.motion_rnn.mlp_layers = 12

"""Train Config"""
# smaller batch size makes loss instable
C.batch_size = 256
C.num_workers = 8

C.cos_lr_max=5e-4
C.cos_lr_mid=3e-4
C.cos_lr_min=1e-5
C.cos_lr_total_iters=115000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 5

"""Display Config"""
C.print_every = 100
C.save_every = 5000

if C.model == 'siMLPe':
	config.pre_dct = True
	config.post_dct = True
	config.residual_output = True
elif C.model == 'Seq2SeqGRU':
	config.pre_dct = False
	config.post_dct = False
	config.residual_output = False
elif C.model == 'siMLPe_RNN':
	config.pre_dct = False
	config.post_dct = False
	config.residual_output = False

if __name__ == '__main__':
    print(config.decoder.motion_mlp)
