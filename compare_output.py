#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np

import argparse

usage_str = 'usage: %(prog)s [options]'
description_str = 'plot ground truth vs prediction'

def tile(ims):
    return np.stack(ims, axis=2)

def imshowtile(x, cmap='gray'):
    plt.imshow(x.transpose((0,2,1)).reshape((x.shape[0], -1)),cmap=cmap)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--slice', action='store', dest='idx', type=int, help='show this slice (Default -- middle)', default=None)
    parser.add_argument('--truth', action='store', dest='npy_truth', type=str, help='ground truth npy file')
    parser.add_argument('--prediction', action='store', dest='npy_predict', type=str, help='prediction npy file')

    args = parser.parse_args()

    if args.idx is None:
        args.idx = data.shape[0] // 2

    z0 = np.load(args.npy_truth)
    z1 = np.load(args.npy_predict)

    z0_idx = z0[args.idx,:,:,:].squeeze().transpose((1,2,0))
    print(z0_idx.shape)
    z1_idx = z1[args.idx,:,:,:]
    print(z1_idx.shape)

    imshowtile(np.concatenate((z0_idx, z1_idx), axis=2))

    plt.show()
