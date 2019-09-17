#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import train
import subtle.subtle_plot as suplot
import subtle.subtle_preprocess as sup
import subtle.subtle_io as suio

import argparse

def tile(ims):
    return np.stack(ims, axis=2)

def imshowtile(x, cmap='gray', vmin=None, vmax=None):
    if not vmax:
        vmin = x.min()
        vmax = x.max()
    plt.imshow(x.transpose((0,2,1)).reshape((x.shape[0], -1)), cmap=cmap, vmin=vmin, vmax=vmax)

def plot_h5(input, output, idx=None, h5_key='data'):
    data = suio.load_file(input, params={'h5_key': h5_key})

    if idx is None:
        idx = data.shape[0] // 2

    X0 = data[idx,0,:,:].squeeze()
    X1 = data[idx,1,:,:].squeeze()
    X2 = data[idx,2,:,:].squeeze()

    plt.figure(figsize=(20,20))
    plt.subplot(3,1,1)
    imshowtile(tile((X0, X1, X2)), vmin=X0.min(), vmax=X0.max())
    plt.colorbar()

    X1mX0 = X1 - X0
    X2mX0 = X2 - X0

    plt.subplot(3,1,2)
    imshowtile(tile((0*X0, X1mX0, X2mX0)))
    plt.colorbar()

    plt.subplot(3,1,3)
    imshowtile(tile((X0, X0 + X1mX0, X0 + X2mX0)), vmin=X0.min(), vmax=X0.max())
    plt.colorbar()

    if output is None:
        plt.show()
    else:
        plt.savefig(output)

usage_str = 'usage: %(prog)s [options]'
description_str = 'plot grid of images from dataset'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--slice', action='store', dest='idx', type=int, help='show this slice (Default -- middle)', default=None)
    parser.add_argument('--input', action='store', dest='input', type=str, help='input npy file')
    parser.add_argument('--output', action='store', dest='output', type=str, help='save output instead of plotting', default=None)
    parser.add_argument('--h5_key', action='store', dest='h5_key', type=str, help='H5 key to get the images from, for plotting', default='data')

    args = parser.parse_args()
    plot_h5(args.input, args.output, args.slice, args.h5_key)
