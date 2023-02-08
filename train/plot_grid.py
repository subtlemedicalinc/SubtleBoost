#!/usr/bin/env python

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.animation import FuncAnimation
from scipy.ndimage.interpolation import rotate

import subtle.subtle_plot as suplot
import subtle.subtle_preprocess as sup
import subtle.utils.io as utils_io

import argparse

def tile(ims):
    return np.stack(ims, axis=2)

def imshowtile(x, cmap='gray', vmin=None, vmax=None):
    if not vmax:
        vmin = x.min()
        vmax = x.max()
    plt.imshow(x.transpose((0,2,1)).reshape((x.shape[0], -1)), cmap=cmap, vmin=vmin, vmax=vmax)

def save_video(input, output, h5_key='data'):
    data = utils_io.load_file(input, params={'h5_key': h5_key})

    X0 = data[:, 0, ...].squeeze()
    X1 = data[:, 1, ...].squeeze()
    X2 = data[:, 2, ...].squeeze()

    X1mX0 = (X1 - X0) * 2
    X2mX0 = (X2 - X0) * 2

    row1 = np.array([np.hstack([x_0, x_1, x_2]) for (x_0, x_1, x_2) in zip(X0, X1, X2)])
    row2 = np.array([np.hstack([x_0, x_1, x_2]) for (x_0, x_1, x_2) in zip(0*X0, X1mX0, X2mX0)])

    skip_count = int(row1.shape[0] * 0.1)

    row1 = row1[skip_count:-skip_count]
    row2 = row2[skip_count:-skip_count]

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

    anim = FuncAnimation(fig, _updatefig, frames=range(row1.shape[0]), interval=75)
    anim.save(output)

def plot_h5(input, output, idx=None, h5_key='data', axis=0):
    data = utils_io.load_file(input, params={'h5_key': h5_key})
    if axis == 0:
        pass
    elif axis == 1:
        data = data.transpose(2, 1, 0, 3)
        data = rotate(data, angle=-90.0, axes=(2, 3))
    elif axis == 2:
        data = data.transpose(3, 1, 0, 2)
        data = rotate(data, angle=-90.0, axes=(2, 3))

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

def get_rgb(img):
    img = (img - np.min(img))/np.ptp(img)
    return np.dstack((img, img, img))

def slice_preview(img_vol, interval=7):
    n_rows = 7
    n_cols = 6
    idx = interval
    all_imgs = []
    bflag = False
    for c in range(n_cols):
        img_rows = []
        for r in range(n_rows):
            if idx >= img_vol.shape[0]:
                img_rows.append(np.zeros_like(img_vol[0]))
                bflag = True
            else:
                img_rows.append(img_vol[idx])
            idx += interval

        all_imgs.append(np.hstack(img_rows))
        if bflag:
            break

    img_disp = np.vstack(all_imgs)
    return img_disp

def plot_multi_contrast(input, output, idx=None, h5_key='data'):
    # if '.h5' in input:
    #     data_all = utils_io.load_file(input, params={'h5_key': h5_key})
    #     data_t1 = utils_io.load_file(input.replace('_T2', ''), params={'h5_key': h5_key})
    # else:
    #     data_full = utils_io.load_file(input)
    #     data_all = data_full[..., 3]
    #     data_t1 = data_full[..., :3]
    #     data_idx = 0 if h5_key == 'data' else 1
    #     data = data_all[data_idx]
    data_all = utils_io.load_file(input, params={'h5_key': 'all'}).astype(np.float32)
    fpath_t1 = input.replace('_T2', '').replace('_FLAIR', '')
    if not os.path.exists(fpath_t1):
        rep = 'h5' if 'npy' in input else 'npy'
        fpath_t1 = fpath_t1.replace(fpath_t1.split('/')[-1].split('.')[-1], rep)
    data_t1 = utils_io.load_file(fpath_t1, params={'h5_key': 'all'}).astype(np.float32)
    data_idx = 0 if h5_key == 'data' else 1
    data = data_all[data_idx]

    plt.figure(figsize=(15, 15))
    mc_disp = slice_preview(data_all[0])
    plt.imshow(mc_disp, cmap='gray')
    plt.axis('off')
    plt.savefig(output)

    csf_th = np.quantile(data_all[0], 0.90)
    csf_mask = data_all[0] >= csf_th
    mc_mask = data_all[1] >= 0.1

    mc_csf = (mc_mask * csf_mask).astype(data_all.dtype)
    t1_base = data_t1[0, :, 0]

    t1_disp = slice_preview(t1_base)
    t1_rgb = get_rgb(t1_disp)
    csf_disp = slice_preview(mc_csf)
    csf_rgb = get_rgb(csf_disp)
    t1_rgb[..., 0] = csf_rgb[..., 0] * 0.5

    disp_scale = 1.2
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.imshow(disp_scale * t1_rgb, vmin=t1_base.min(), vmax=t1_base.max())
    plt.axis('off')
    plt.savefig(output.replace('.png', '_csf.png'))

usage_str = 'usage: %(prog)s [options]'
description_str = 'plot grid of images from dataset'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--slice', action='store', dest='idx', type=int, help='show this slice (Default -- middle)', default=None)
    parser.add_argument('--axis', action='store', dest='axis', type=int, help='reformat to axis', default=0)
    parser.add_argument('--input', action='store', dest='input', type=str, help='input npy file')
    parser.add_argument('--output', action='store', dest='output', type=str, help='save output instead of plotting', default=None)
    parser.add_argument('--h5_key', action='store', dest='h5_key', type=str, help='H5 key to get the images from, for plotting', default='data')
    parser.add_argument('--multi_contrast', action='store_true', default=False)
    parser.add_argument('--video', action='store_true', dest='video', help='If true, the preprocessed and difference is stored as an MP4 video', default=False)

    args = parser.parse_args()
    if args.multi_contrast:
        plot_multi_contrast(args.input, args.output, args.idx, args.h5_key)
    else:
        plot_h5(args.input, args.output, args.idx, args.h5_key, args.axis)

    if args.video:
        save_video(args.input, args.output, args.h5_key)
