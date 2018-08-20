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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--slice', action='store', dest='idx', type=int, help='show this slice (Default -- middle)', default=None)
    parser.add_argument('--truth', action='store', dest='file_truth', type=str, help='ground truth file')
    parser.add_argument('--prediction', action='store', dest='file_predict', type=str, help='prediction file')
    parser.add_argument('--output', action='store', dest='output', type=str, help='save output instead of plotting', default=None)

    args = parser.parse_args()

    z0 = suio.load_file(args.file_truth)
    z1 = suio.load_file(args.file_predict)

    if args.idx is None:
        args.idx = z0.shape[0] // 2

    z0_idx = z0[args.idx,:,:,:].squeeze().transpose((1,2,0))
    z1_idx = z1[args.idx,:,:][:,:,None]

    plt.figure(figsize=(20,5))
    imshowtile(np.concatenate((z0_idx, z1_idx), axis=2))
    plt.title('pre vs. low vs. full vs. predicted')

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output)
