#!/usr/bin/env python

import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import argparse

import subtle.utils.io as utils_io
import subtle.subtle_metrics as sumetrics

usage_str = 'usage: %(prog)s [options]'
description_str = 'compute performance stats from data'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--all_slices', action='store_true', dest='all_slices',  help='show all slices', default=False)
    parser.add_argument('--slice', action='append', dest='idxs', type=int, help='show this slice (Default -- middle)', default=[])
    parser.add_argument('--truth', action='store', dest='file_truth', type=str, help='ground truth file')
    parser.add_argument('--prediction', action='store', dest='file_predict', type=str, help='prediction file')
    parser.add_argument('--cutoff', action='store', dest='cutoff', type=float, help='cut off first and last percent of slices', default=0.)
    parser.add_argument('--output', action='store', dest='output_file', type=str, help='output h5 file', default=None)

    args = parser.parse_args()

    data_truth = utils_io.load_file(args.file_truth)
    data_predict = utils_io.load_file(args.file_predict)

    n_slices = data_truth.shape[0]

    print("Number of slices:", n_slices)
    print()

    if args.all_slices:
        args.idxs = np.arange(int(n_slices * args.cutoff), int(n_slices * (1 - args.cutoff)))

    stats = {'pred/nrmse': [], 'pred/psnr': [], 'pred/ssim': [], 'low/nrmse': [], 'low/psnr': [], 'low/ssim': []}

    if len(args.idxs) == 0:
        args.idxs = [n_slices // 2]

    for idx in sorted(args.idxs):
        if idx < n_slices and idx >= 0:
            x_zero = data_truth[idx,-1,:,:].squeeze()
            x_low = data_truth[idx,-2,:,:].squeeze()
            x_pred = data_predict[idx,:,:].squeeze()

            stats['low/nrmse'].append(sumetrics.nrmse(x_zero, x_low))
            stats['low/ssim'].append(sumetrics.ssim(x_zero, x_low))
            stats['low/psnr'].append(sumetrics.psnr(x_zero, x_low))

            stats['pred/nrmse'].append(sumetrics.nrmse(x_zero, x_pred))
            stats['pred/ssim'].append(sumetrics.ssim(x_zero, x_pred))
            stats['pred/psnr'].append(sumetrics.psnr(x_zero, x_pred))

    with h5py.File(args.output_file, 'w') as f:
        for key in stats.keys():
            f.create_dataset(key, data=stats[key])

    print('stat\tmean\tstdev')
    print('LOW NRMSE\t{}\t{}'.format(np.mean(stats['low/nrmse']), np.std(stats['low/nrmse'])))
    print('PRED NRMSE\t{}\t{}'.format(np.mean(stats['pred/nrmse']), np.std(stats['pred/nrmse'])))
    print('LOW PSNR\t{}\t{}'.format(np.mean(stats['low/psnr']), np.std(stats['low/psnr'])))
    print('PRED PSNR\t{}\t{}'.format(np.mean(stats['pred/psnr']), np.std(stats['pred/psnr'])))
    print('LOW SSIM\t{}\t{}'.format(np.mean(stats['low/ssim']), np.std(stats['low/ssim'])))
    print('PRED SSIM\t{}\t{}'.format(np.mean(stats['pred/ssim']), np.std(stats['pred/ssim'])))
