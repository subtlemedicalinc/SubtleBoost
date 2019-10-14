#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.animation import FuncAnimation

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

def save_video(input, output, h5_key='data'):
    data = suio.load_file(input, params={'h5_key': h5_key})

    X0 = data[:, 0, ...].squeeze()
    X1 = data[:, 1, ...].squeeze()
    X2 = data[:, 2, ...].squeeze()

    X1mX0 = (X1 - X0) * 2
    X2mX0 = (X2 - X0) * 2

    row1 = np.array([np.hstack([x_0, x_1, x_2]) for (x_0, x_1, x_2) in zip(X0, X1, X2)])
    row2 = np.array([np.hstack([x_0, x_1, x_2]) for (x_0, x_1, x_2) in zip(0*X0, X1mX0, X2mX0)])
    out = np.array([np.vstack([r1, r2]) for r1, r2 in zip(row1, row2)])

    outfile = output.replace('png', 'mp4')

    fig = plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    im1 = plt.imshow(row1[0], cmap='gray', vmin=data[:, 0].min(), vmax=data[:, 0].max())
    plt.axis('off')

    plt.subplot(2, 1, 2)
    im2 = plt.imshow(row2[0], cmap='gray', vmin=data[:, 0].min(), vmax=data[:, 0].max())
    plt.axis('off')

    fig.tight_layout()

    def _updatefig(idx):
        im1.set_data(row1[idx])
        im2.set_data(row2[idx])
        return [im1, im2]

    anim = FuncAnimation(fig, _updatefig, frames=range(out.shape[0]), interval=50)
    anim.save(outfile)


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
    parser.add_argument('--video', action='store_true', dest='video', help='If true, the preprocessed and difference is stored as an MP4 video', default=False)

    args = parser.parse_args()
    plot_h5(args.input, args.output, args.idx, args.h5_key)

    if args.video:
        save_video(args.input, args.output, args.h5_key)
