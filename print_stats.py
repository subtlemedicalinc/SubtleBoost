#!/usr/bin/env python

import sys

import h5py
import numpy as np
import argparse

usage_str = 'usage: %(prog)s [options]'
description_str = 'print stats from h5 stats file'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--all_output', action='store_true', dest='all_output', help='print output for each file separately', default=False)
    parser.add_argument('inputs',  type=str, nargs='+', help='show all slices', default=False)
    args = parser.parse_args()

    stats = {'pred/nrmse': [], 'pred/psnr': [], 'pred/ssim': [], 'low/nrmse': [], 'low/psnr': [], 'low/ssim': []}

    for input in args.inputs:

        if args.all_output:
            print('{}:'.format(input))

        with h5py.File(input, 'r') as F:
            for key in stats.keys():
                val = np.array(F[key])[0]
                stats[key].append(val)
                print('{:>10}: {:10.5f}'.format(key, val))
        print()

    print('{:>10}\t{:>10}\t{:>10}'.format('STAT', 'MEAN', 'STDEV'))

    print('{:>10}\t{:10.5f}\t{:10.5f}'.format('LOW NRMSE', np.mean(stats['low/nrmse']), np.std(stats['low/nrmse'])))
    print('{:>10}\t{:10.5f}\t{:10.5f}'.format('PRED NRMSE', np.mean(stats['pred/nrmse']), np.std(stats['pred/nrmse'])))

    print('{:>10}\t{:10.5f}\t{:10.5f}'.format('LOW PSNR', np.mean(stats['low/psnr']), np.std(stats['low/psnr'])))
    print('{:>10}\t{:10.5f}\t{:10.5f}'.format('PRED PSNR', np.mean(stats['pred/psnr']), np.std(stats['pred/psnr'])))

    print('{:>10}\t{:10.5f}\t{:10.5f}'.format('LOW SSIM', np.mean(stats['low/ssim']), np.std(stats['low/ssim'])))
    print('{:>10}\t{:10.5f}\t{:10.5f}'.format('PRED SSIM', np.mean(stats['pred/ssim']), np.std(stats['pred/ssim'])))
