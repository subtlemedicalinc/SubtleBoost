'''
subtle_plot.py

Visualization functions for contrast synthesis
Based on SimpleITK examples

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/18
'''

import sys
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt

import subtle.subtle_metrics as sumetrics

try:
    sys.path.insert(0, '/home/subtle/jon/tools/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.5/SimpleITK')
    import SimpleITK as sitk
except:
    warnings.warn('SimpleITK not found!')
import scripts.utils.myshow as myshow

def imshow3(ims, axis=0, cmap='gray'):
    n = ims.shape[axis]
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(np.take(ims, i, axis=axis), cmap=cmap)

def simshow(sitk_image, cmap='gray'):
    plt.imshow(sitk.GetArrayViewFromImage(sitk_image), cmap=cmap)

def imshowreg(im0, im1, title=None):
    im0n = (im0 - np.min(im0))
    im1n = (im1 - np.min(im1))
    im0n = im0n / np.max(im0n)
    im1n = im1n / np.max(im1n)
    im = np.stack((im0n, im1n, 0*im1n), axis=2)
    plt.subplot(1,2,1)
    plt.imshow(im)
    if title is not None:
        plt.title('{} (overlay)'.format(title))
    plt.subplot(1,2,2)
    plt.imshow(np.abs(im0n - im1n))
    if title is not None:
        plt.title('{} (diff)'.format(title))

def myshow3d(im, title=None, margin=0.05, dpi=80, figsize=None ):
    myshow.myshow3d(sitk.GetImageFromArray(im), title=title, margin=margin, dpi=dpi, figsize=figsize)

def imshowcmp(im0, im1, title='', figsize=None):
    im0n = (im0 - np.min(im0))
    im1n = (im1 - np.min(im1))
    im0n = im0n / np.max(im0n)
    im1n = im1n / np.max(im1n)
    im = np.stack((im0n, im1n, 0*im1n), axis=3)
    myshow3d(im, title=title, figsize=figsize)

def tile(ims):
    return np.stack(ims, axis=2)

def imshowtile(x, title=None, cmap='gray'):
    plt.imshow(x.transpose((0,2,1)).reshape((x.shape[0], -1)),cmap=cmap)

def compare_output(data_truth, data_predict, idx=None, show_diff=False, output=None):
    plt.switch_backend('agg')

    if idx is None:
        idx = data_truth.shape[0] // 2

    data_truth_idx = data_truth[idx,:,:,:].squeeze().transpose((1,2,0)).astype(np.float64)
    data_predict_idx = data_predict[idx,:,:].squeeze()[:,:,None].astype(np.float64)

    nrmse = sumetrics.nrmse(data_truth_idx[:,:,-1], data_predict_idx.squeeze())
    psnr = sumetrics.psnr(data_truth_idx[:,:,-1], data_predict_idx.squeeze())
    ssim = sumetrics.ssim(data_truth_idx[:,:,-1], data_predict_idx.squeeze())

    metrics_text = '\n'.join((
        'NRMSE = {:.3f}'.format(nrmse),
        'PSNR  = {:.3f}'.format(psnr),
        'SSIM  = {:.3f}'.format(ssim)))

    print('Slice {}'.format(idx))
    print(metrics_text)
    print()

    plt.figure(figsize=(20,5))
    if show_diff:
        data_truth_idx_diff = abs(data_truth_idx[:,:,2] - data_truth_idx[:,:,0])
        data_predict_idx_diff = abs(data_predict_idx[:,:,0] - data_truth_idx[:,:,0])
        imshowtile(np.concatenate((data_truth_idx[:,:,0][:,:,None], data_truth_idx_diff[:,:,None], data_truth_idx[:,:,2][:,:,None], data_predict_idx_diff[:,:,None], data_predict_idx), axis=2))
        plt.title('pre vs. truth diff vs. truth vs. SubtleGad diff vs. SubtleGad')
    else:
        imshowtile(np.concatenate((data_truth_idx, data_predict_idx), axis=2))
        plt.title('Pre-contrast vs. 10% Dose vs. Post-contrast vs. SubtleGad')
        ax = plt.gca()
        props = dict(boxstyle='round', facecolor='wheat', alpha=1.)
        ax.text(1.01, 0.95, metrics_text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)


    if output is not None:
        plt.savefig(output)

    return nrmse, psnr, ssim
