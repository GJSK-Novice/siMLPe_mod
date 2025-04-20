# Code to run in bash console
# cd exps/baseline_amass
# %load_ext autoreload
# %autoreload 2

import argparse
import os, sys
import json
import math
import numpy as np
import copy

from config import config
import model as models
from datasets.amass import AMASSDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir

from custom_test import test
from datasets.amass_eval import AMASSEval

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# cuda setting to make result deterministic
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
# default is False for 'store_true'
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

# args = parser.parse_args()
# pass argument without command line
import shlex
argString = '--seed 888 --exp-name baseline.txt --layer-norm-axis spatial --with-normalization --num 48'
args = parser.parse_args(shlex.split(argString))

torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

# config.motion_rnn.with_normalization = args.with_normalization

acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))

def get_dct_matrix(N):
	dct_m = np.eye(N)
	for k in np.arange(N):
		for i in np.arange(N):
			w = np.sqrt(2 / N)
			if k == 0:
				w = np.sqrt(1 / N)
			dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
	idct_m = np.linalg.inv(dct_m)
	return dct_m, idct_m

# size: (1,T,T)
if config.pre_dct:
	dct_m,idct_m = get_dct_matrix(config.motion.amass_input_length_dct)
	dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
	idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, mid_lr, min_lr, optimizer):
	if nb_iter < 50000:
		current_lr = max_lr
	elif nb_iter < 100000:
		current_lr = mid_lr
	else:
		current_lr = min_lr

	for param_group in optimizer.param_groups:
		param_group["lr"] = current_lr

	return optimizer, current_lr

def gen_velocity(m):
	dm = m[:, 1:] - m[:, :-1]
	return dm

def train_step(amass_motion_input, amass_motion_target, model, optimizer, nb_iter, total_iter, max_lr, mid_lr, min_lr) :

	if config.pre_dct:
		b,n,c = amass_motion_input.shape
		amass_motion_input_ = amass_motion_input.clone()
		amass_motion_input_ = torch.matmul(dct_m, amass_motion_input_.cuda())
	else:
		amass_motion_input_ = amass_motion_input.clone()

	motion_pred = model(amass_motion_input_.cuda())

	if config.post_dct:
		motion_pred = torch.matmul(idct_m, motion_pred)

	if config.residual_output:
		offset = amass_motion_input[:, -1:].cuda()
		motion_pred = motion_pred[:, :config.motion.amass_target_length] + offset
	else:
		motion_pred = motion_pred[:, :config.motion.amass_target_length]

	# calc losses
	b,n,c = amass_motion_target.shape
	motion_pred = motion_pred.reshape(b,n,18,3).reshape(-1,3)
	amass_motion_target = amass_motion_target.cuda().reshape(b,n,18,3).reshape(-1,3)
	loss = torch.mean(torch.norm(motion_pred - amass_motion_target, 2, 1))
	# add position loss and velocity loss
	if config.use_relative_loss:
		motion_pred = motion_pred.reshape(b,n,18,3)
		dmotion_pred = gen_velocity(motion_pred)
		motion_gt = amass_motion_target.reshape(b,n,18,3)
		dmotion_gt = gen_velocity(motion_gt)
		dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
		loss = loss + dloss
	else:
		loss = loss.mean()

	writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

	# reset gradients
	optimizer.zero_grad()
	# compute gradients by backpropagation
	loss.backward()
	# update params
	optimizer.step()
	optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, mid_lr, min_lr, optimizer)
	writer.add_scalar('LR/train', current_lr, nb_iter)

	return loss.item(), optimizer, current_lr

if config.model == 'siMLPe':
	model = models.siMLPe(config)
elif config.model == 'siMLPe_RNN':
	model = models.SlidingRNN_v2(config)
elif config.model == 'Seq2SeqGRU':
	model = models.Seq2SeqGRU(config)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print()
print("Total count of parameters:",total_params)
print("RNN type? ","GRU" if config.motion_rnn.use_gru else "LSTM")
print("Residual output? ",config.residual_output)
print("Use DCT? ",config.pre_dct, config.post_dct)
print("Using recursive residual?",config.motion_rnn.recursive_residual)
# print("Using LayerNorm?",config.motion_rnn.with_normalization) (deprecated)
print("Using spatial fc before temporal in RNN?",config.motion_rnn.local_spatial_fc)
print("Temporal layer in RNN:",config.motion_rnn.num_temp_blocks)
# print("Sliding long term encoder in RNN? ",config.motion_rnn.sliding_long_term) (deprecated)
print("History term window size: ",config.motion_rnn.history_window_size)
print("Short term window size: ",config.motion_rnn.short_term_window_size)
print("Encode history? ",config.motion_rnn.encode_history)
print("mlp_layers = ",config.motion_rnn.mlp_layers)
print("rnn_state_size = ",config.motion_rnn.rnn_state_size)
print("rnn_layers = ",config.motion_rnn.rnn_layers)
print("rnn_blocks = ",config.motion_rnn.rnn_blocks)

model.train().cuda()

# dataset = (T-by-C x_in, N-by-C x_out)
config.motion.amass_target_length = config.motion.amass_target_length_train
dataset = AMASSDataset(config, 'train', config.data_aug)

# separate into batches (input, target) with size (batch_size,T,C) and (batch_size,N,C)
shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
						num_workers=config.num_workers, drop_last=True,
						sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_config = copy.deepcopy(config)
eval_config.motion.amass_target_length = eval_config.motion.amass_target_length_eval
eval_dataset = AMASSEval(eval_config, 'test')

shuffle = False
sampler = None
# separate into batches (input, target) with size (batch_size,T=50,K,3) and (batch_size,N=25,K,3)
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
						num_workers=1, drop_last=False,
						sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
							 lr=config.cos_lr_max,
							 weight_decay=config.weight_decay)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

# continue training from a checkpoint
if config.model_pth is not None:
	state_dict = torch.load(config.model_pth)
	model.load_state_dict(state_dict, strict=True)
	print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))
	print("Loading model path from {} ".format(config.model_pth))

# Training
nb_iter = 0
avg_loss = 0
avg_lr = 0
current_lr = config.cos_lr_max

config.save_every = 5000
config.cos_lr_total_iters = 5000
# siMLPe_results = [10.8, 19.6, 34.3, 40.5, 50.5, 57.3, 62.4, 65.7]
# baseline_results = [23.8,44.4,76.1,88.2,107.4,121.6,131.6,136.6]
# our_best_results = [11.2, 25.1, 51.7, 63.2, 82.0, 96.3, 108.3, 116.1]

with tqdm(total=config.cos_lr_total_iters, desc="Training on AMASS") as pbar:
	while (nb_iter + 1) < config.cos_lr_total_iters:
		for (amass_motion_input, amass_motion_target) in dataloader:

			loss, optimizer, current_lr = train_step(amass_motion_input, amass_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_mid, config.cos_lr_min)
			avg_loss += loss
			avg_lr += current_lr

			if (nb_iter + 1) % config.print_every ==  0 :
				avg_loss = avg_loss / config.print_every
				avg_lr = avg_lr / config.print_every

				print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
				print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
				avg_loss = 0
				avg_lr = 0

			# if (nb_iter + 1) % config.save_every ==  0 :
				# if (nb_iter + 1) > config.cos_lr_total_iters - config.save_every - 1:
				# 	torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
				# model.eval()
				# acc_tmp = test(eval_config, model, eval_dataloader)
				# print(f'Iteration {nb_iter + 1} results: {", ".join(str(i) for i in acc_tmp)}')
				# print([round(float(acc_tmp[i]-siMLPe_results[i]),2) for i in range(8)])
				# acc_log.write(f"{nb_iter + 1}\n{' '.join(str(a) for a in acc_tmp)}\n")
				# model.train()

			if (nb_iter + 1) == config.cos_lr_total_iters :
				break
			nb_iter += 1
		pbar.update(nb_iter - pbar.n)
torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
writer.close()
