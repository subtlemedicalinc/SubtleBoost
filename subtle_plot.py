'''
subtle_plot.py

Visualization functions for contrast synthesis
Based on SimpleITK examples

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/18
'''

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/subtle/jon/tools/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.5/SimpleITK')
import SimpleITK as sitk
import myshow

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