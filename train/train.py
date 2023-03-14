#!/usr/bin/env python
'''
train.py

Training for contrast synthesis.
Trains netowrk using dataset of npy files

@author: Srivathsa Pasumarthi (srivathsa@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2023/03/13
'''

import os
import datetime
import time
import random
from warnings import warn
import configargparse as argparse
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np

from subtle.dnn.helpers import load_model, load_db_class, AverageMeter

import subtle.utils.io as utils_io
import subtle.utils.experiment as utils_exp
import subtle.utils.hyperparameter as utils_hyp
from subtle.data_loaders import SliceLoader
import subtle.subtle_loss as suloss
import subtle.subtle_metrics as sumetrics
import subtle.subtle_plot as suplot
from subtle.dnn.callbacks import plot_tb

usage_str = 'usage: %(prog)s [options]'
description_str = 'Train SubtleGrad network on pre-processed data.'

def train_process(args):
    print('------')
    print(args)
    print('------\n\n\n')

    if args.max_data_sets is None:
        max_data_sets = np.inf
    else:
        max_data_sets = args.max_data_sets

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    np.random.seed(args.random_seed)

    case_nums = utils_exp.get_experiment_data(args.experiment, dataset='train')
    data_list = ['{}/{}'.format(args.data_dir, cnum) for cnum in case_nums]

    fpath_sample = [
        f for f in glob('{}/{}/**/*.npy'.format(args.data_dir, case_nums[0]), recursive=True)
    ][0]
    data_sample = np.load(fpath_sample)
    _, _, nx, ny = data_sample.shape

    args.input_idx = [int(idx) for idx in args.input_idx.split(',')]
    args.output_idx = [int(idx) for idx in args.output_idx.split(',')]
    model_class = load_model(args)

    model_kwargs = {
        'model_config': args.model_config,
        'num_channel_output': len(args.output_idx) if not args.multi_slice_gt else args.slices_per_input,
        'verbose': args.verbose
    }

    G = model_class(**model_kwargs)

    tic = time.time()

    if len(data_list) == 1 or args.validation_split == 0:
        r = 0
    else: # len(data_list) > 1
        r = int(len(data_list) * args.validation_split)

    val_data = utils_exp.get_experiment_data(args.experiment, dataset='val')
    if len(val_data) > 0:
        data_val_list = ['{}/{}'.format(args.data_dir, v) for v in val_data]
        data_train_list = data_list
    else:
        data_val_list = data_list[:r]
        data_train_list = data_list[r:]

    if args.verbose:
        print('using {} datasets for training:'.format(len(data_train_list)))
        for d in data_train_list:
            print(d)
        print('using {} datasets for validation:'.format(len(data_val_list)))
        for d in data_val_list:
            print(d)

    num_epochs = args.num_epochs
    db_class = load_db_class(args)

    gen_kwargs = {
        'data_files': data_train_list,
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'verbose': args.verbose,
        'slices_per_input': args.slices_per_input,
        'input_idx': args.input_idx,
        'output_idx': args.output_idx,
        'resize': args.resize,
        'resample_size': args.resample_size,
        'use_enh_mask': args.enh_mask,
        'enh_pfactor': args.enh_pfactor,
        'file_ext': args.file_ext
    }

    train_db = db_class(**gen_kwargs)

    gen_kwargs['data_files'] = data_val_list
    gen_kwargs['shuffle'] = False
    val_db = db_class(**gen_kwargs)

    def worker_init_fn(worker_id):
        random.seed(args.random_seed)

    train_loader = DataLoader(
        train_db, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn
    )
    opt_G = Adam(G.parameters(), lr=args.lr_init, amsgrad=args.optim_amsgrad)

    for epoch_num in np.arange(num_epochs):
        G.train()

        l_total = AverageMeter()
        l_l1 = AverageMeter()
        l_ssim = AverageMeter()

        for i_batch, data in enumerate(tqdm(train_loader)):
            X, Y = data
            Y_pred = G(X)

            l1 = suloss.l1_loss(Y[:, 0], Y_pred[:, 0])

            opt_G.zero_grad()
            l1.backward()
            opt_G.step()

            break
        break
