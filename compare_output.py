#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import argparse

import subtle.subtle_io as suio

usage_str = 'usage: %(prog)s [options]'
description_str = 'plot ground truth vs prediction'

def tile(ims):
    return np.stack(ims, axis=2)

def imshowtile(x, title=None, cmap='gray'):
    plt.imshow(x.transpose((0,2,1)).reshape((x.shape[0], -1)),cmap=cmap)

def make_plot(data_truth, data_predict, idx=None, show_diff=False, output=None):
    if idx is None:
        idx = data_truth.shape[0] // 2

    data_truth_idx = data_truth[idx,:,:,:].squeeze().transpose((1,2,0))
    data_predict_idx = data_predict[idx,:,:].squeeze()[:,:,None]

    plt.figure(figsize=(20,5))
    if show_diff:
        data_truth_idx_diff = abs(data_truth_idx[:,:,2] - data_truth_idx[:,:,0])
        data_predict_idx_diff = abs(data_predict_idx[:,:,0] - data_truth_idx[:,:,0])
        imshowtile(np.concatenate((data_truth_idx[:,:,0][:,:,None], data_truth_idx_diff[:,:,None], data_truth_idx[:,:,2][:,:,None], data_predict_idx_diff[:,:,None], data_predict_idx), axis=2))
        plt.title('pre vs. truth diff vs. truth vs. SubtleGad diff vs. SubtleGad')
    else:
        imshowtile(np.concatenate((data_truth_idx, data_predict_idx), axis=2))
        plt.title('pre vs. low vs. full vs. predicted')

    if output is None:
        plt.show()
    else:
        plt.savefig(output)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--slice', action='store', dest='idx', type=int, help='show this slice (Default -- middle)', default=None)
    parser.add_argument('--truth', action='store', dest='file_truth', type=str, help='ground truth file')
    parser.add_argument('--prediction', action='store', dest='file_predict', type=str, help='prediction file')
    parser.add_argument('--output', action='store', dest='output', type=str, help='save output instead of plotting', default=None)
    parser.add_argument('--show_diff', action='store_true', dest='show_diff', help='show diff images', default=False)

    args = parser.parse_args()

    data_truth = suio.load_file(file_truth)
    data_predict = suio.load_file(file_predict)

    make_plot(data_truth, data_predict, idx=args.idx, show_diff=args.show_diff, output=args.output)

