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

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter

import numpy as np

from subtle.dnn.helpers import load_model, load_db_class, AverageMeter, make_image_grid

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

def eval_model(args, model, eval_db):
    def worker_init_fn(worker_id):
        random.seed(args.random_seed)

    metrics = {
        'psnr': AverageMeter(),
        'ssim': AverageMeter(),
        'mse': AverageMeter()
    }

    val_loader = DataLoader(
        eval_db, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=False, worker_init_fn=worker_init_fn
    )

    model.eval()
    metrics_freq = 25
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(val_loader)):
            if i_batch % metrics_freq == 0:
                X, Y = data
                X = X.to('cuda')
                Y = Y.to('cuda')

                Y_pred = model(X)[:, 0].detach().cpu().numpy()
                Y_true = Y[:, 0].detach().cpu().numpy()

                metrics['psnr'].update(sumetrics.psnr(Y_true, Y_pred))
                metrics['ssim'].update(sumetrics.ssim(Y_true, Y_pred))
                metrics['mse'].update(sumetrics.nrmse(Y_true, Y_pred))

    return {
        'psnr': metrics['psnr'].avg,
        'ssim': metrics['ssim'].avg,
        'mse': metrics['mse'].avg
    }

def train_process(args):
    print('------')
    print(args)
    print('------\n\n\n')

    t1 = time.time()
    torch.set_num_threads(8)

    if args.max_data_sets is None:
        max_data_sets = np.inf
    else:
        max_data_sets = args.max_data_sets

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    case_nums = utils_exp.get_experiment_data(args.experiment, dataset='train')
    data_list = ['{}/{}'.format(args.data_dir, cnum) for cnum in case_nums]

    fpath_sample = [
        f for f in glob('{}/{}/**/*.npy'.format(args.data_dir, case_nums[0]), recursive=True)
    ][0]
    data_sample = np.load(fpath_sample)
    _, nx, ny = data_sample.shape

    args.input_idx = [int(idx) for idx in args.input_idx.split(',')]
    args.output_idx = [int(idx) for idx in args.output_idx.split(',')]
    model_class = load_model(args)

    model_kwargs = {
        'model_config': args.model_config,
        'num_channel_output': len(args.output_idx) if not args.multi_slice_gt else args.slices_per_input,
        'num_channel_input': (args.slices_per_input * 2),
        'verbose': args.verbose,
        'img_rows': nx,
        'img_cols': ny
    }

    G = model_class(**model_kwargs)
    G = G.to('cuda')

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

    if args.data_batch is not None:
        idx1, idx2 = [int(s) for s in args.data_batch.split(',')]
        data_train_list = data_train_list[idx1:idx2]

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
        'file_ext': args.file_ext,
        'slice_axis': [0, 2, 3] if args.train_mpr else [2]
    }

    train_db = db_class(**gen_kwargs)

    plot_cases = utils_exp.get_experiment_data(args.experiment, dataset='plot')
    plane_map = {'ax': 0, 'sag': 2, 'cor': 3}
    gen_kwargs['data_files'] = data_val_list
    gen_kwargs['slice_axis'] = [plane_map[plot_cases[0][1]]]
    gen_kwargs['shuffle'] = False
    val_db = db_class(**gen_kwargs)

    def worker_init_fn(worker_id):
        random.seed(args.random_seed)

    train_loader = DataLoader(
        train_db, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn
    )

    opt_G = Adam(G.parameters(), lr=args.lr_init, amsgrad=args.optim_amsgrad, eps=1e-7)

    tstr = str(time.time()).split('.')[0]
    dpath_log = os.path.join(args.log_dir, args.checkpoint_dir.split('/')[-1])
    iter_num = 0

    best_mse = np.inf
    best_psnr = 0
    best_ssim = 0
    start_epoch = 0

    if args.resume_from_checkpoint:
        print('Resuming from checkpoint - {}'.format(args.resume_from_checkpoint))

        args.checkpoint_dir = os.path.join(
            '/'.join(args.checkpoint_dir.split('/')[:-1]), args.resume_from_checkpoint
        )
        dpath_log = os.path.join(
            '/'.join(dpath_log.split('/')[:-1]), args.resume_from_checkpoint
        )

        fpath_ckps = [fp for fp in glob('{}/epoch*.pth'.format(args.checkpoint_dir))]
        fpath_ckps = sorted(
            fpath_ckps,
            key=lambda fp: int(fp.split('/')[-1].replace('epoch_', '').replace('.pth', ''))
        )
        state_dict = torch.load(fpath_ckps[-1], map_location='cpu')
        iter_num = int(state_dict['opt_G']['state'][0]['step'])
        start_epoch = state_dict['epoch'] + 1

        G.load_state_dict(state_dict['G'])
        opt_G.load_state_dict(state_dict['opt_G'])

        best_mse = state_dict['mse']
        best_psnr = state_dict['psnr']
        best_ssim = state_dict['ssim']

    tb_writer = SummaryWriter(dpath_log)

    flist_plot = [
        f'{args.data_dir}/{cnum}/{plane}/{sl_idx:03d}.{args.file_ext}'.format(sl_idx)
        for cnum, plane, sl_idx in plot_cases
    ]

    plotX = []
    plotY = []
    plotEmask = []

    for fpath in flist_plot:
        pX, pY = val_db.__getitem__(fpath=fpath)
        plotX.append(pX)
        plotY.append(pY[0])
        plotEmask.append(pY[1])

    plotX = torch.from_numpy(np.array(plotX).astype(np.float32)).to('cuda')
    plotY = torch.from_numpy(np.array(plotY).astype(np.float32)).to('cuda')
    plotEmask = torch.from_numpy(np.array(plotEmask).astype(np.float32)).to('cuda')

    vgg_loss = None
    if args.perceptual_lambda > 0:
        vgg_loss = suloss.VGGLoss(
            fpath_ckp=args.vgg19_ckp, img_resize=args.vgg_resize_shape
        ).to('cuda')

    for epoch_num in np.arange(start_epoch, num_epochs):
        G.train()

        l_total = AverageMeter()
        l_l1 = AverageMeter()
        l_ssim = AverageMeter()

        pbar = tqdm(train_loader)
        for i_batch, data in enumerate(pbar):
            iter_num += 1

            X, Y = data
            X = X.to('cuda')
            Y = Y.to('cuda')

            # print('X', X.min(), X.max(), 'Y', Y.min(), Y.max())
            Y_pred = G(X)
            # print('Y pred', Y_pred.min().item(), Y_pred.max().item())

            loss_g, indiv_loss = suloss.mixed_loss(args, Y, Y_pred, vgg_loss)
            # indiv_loss is not scaled with lambda

            mean_total = torch.mean(loss_g).item()
            mean_l1 = torch.mean(indiv_loss['l1']).item()
            mean_ssim = indiv_loss['ssim'].item()

            l_total.update(mean_total)
            l_l1.update(mean_l1)
            l_ssim.update(mean_ssim)

            pbar.set_description(
                f'Total weighted loss: {mean_total:03f}, L1: {mean_l1:03f}'
            )

            tb_writer.add_scalar('train/loss_total', mean_total, iter_num)
            tb_writer.add_scalar('train/loss_l1', mean_l1, iter_num)
            tb_writer.add_scalar(
                'train/loss_ssim', mean_ssim, iter_num
            )
            opt_G.zero_grad()
            loss_g.sum().backward()
            opt_G.step()

        plotY_pred = G(plotX)[:, 0]

        for p_idx in np.arange(plotX.shape[0]):
            h = args.slices_per_input // 2
            preX = plotX[p_idx, :args.slices_per_input]
            lowX = plotX[p_idx, args.slices_per_input:]

            plot_img = make_image_grid([
                preX[h], lowX[h], plotY[p_idx], plotEmask[p_idx], plotY_pred[p_idx]
            ])

            tb_writer.add_image('Validation {}'.format(p_idx), plot_img, epoch_num)

        tb_writer.add_scalar('epoch/loss_total', l_total.avg, epoch_num)
        tb_writer.add_scalar('epoch/loss_l1', l_l1.avg, epoch_num)
        tb_writer.add_scalar('epoch/loss_ssim', l_ssim.avg, epoch_num)

        metrics = eval_model(args, G, val_db)
        tb_writer.add_scalar('epoch/metrics_psnr', metrics['psnr'], epoch_num)
        tb_writer.add_scalar('epoch/metrics_ssim', metrics['ssim'], epoch_num)
        tb_writer.add_scalar('epoch/metrics_mse', metrics['mse'], epoch_num)
        print('Epoch #{}:'.format(epoch_num), metrics)

        state_dict = {
            'G': G.state_dict(),
            'epoch': epoch_num,
            'opt_G': opt_G.state_dict()
        }

        if metrics['mse'] < best_mse:
            best_mse = metrics['mse']
            state_dict['mse'] = best_mse
            fpath_ckp = os.path.join(args.checkpoint_dir, 'best_mse.pth')
            torch.save(state_dict, fpath_ckp)

        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            state_dict['psnr'] = best_psnr
            fpath_ckp = os.path.join(args.checkpoint_dir, 'best_psnr.pth')
            torch.save(state_dict, fpath_ckp)

        if metrics['ssim'] > best_ssim:
            best_ssim = metrics['ssim']
            state_dict['ssim'] = best_ssim
            fpath_ckp = os.path.join(args.checkpoint_dir, 'best_ssim.pth')
            torch.save(state_dict, fpath_ckp)

        state_dict = {**state_dict, **metrics}
        fpath_ckp = os.path.join(args.checkpoint_dir, 'epoch_{}.pth'.format(epoch_num))
        torch.save(state_dict, fpath_ckp)
    tb_writer.close()

    t2 = time.time()
    print('Training finished in {} secs'.format(t2 - t1))
