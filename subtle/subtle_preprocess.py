'''
subtle_preprocess.py

Pre-processing utilities for contrast synthesis
Mask, intensity normalization, and image co-registration

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/18
'''

import sys

import time

import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

sys.path.insert(0, '/home/subtle/jon/tools/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.5/SimpleITK')
import SimpleITK as sitk
    
    
def mask_im(im, threshold=.08):
    '''
    Image masking
    Masks an image set based on max val compared to threshold.
    Fills holes in main mask
    '''
    N, n, nx, ny = im.shape
    mask = im > (threshold * np.amax(im, axis=(2,3))[:,:,None,None])
    # fill holes in mask
    mask = binary_fill_holes(mask.reshape((n*N*nx, ny))).reshape((N, n, nx, ny))
    return mask   
    
def scale_im(im_fixed, im_moving, levels=1024, points=7, mean_intensity=True, verbose=True):
    '''
    Image intensity normalization using SimpleITK
    Normalize im_moving to match im_fixed
    '''
    
    sim0 = sitk.GetImageFromArray(im_fixed)
    sim1 = sitk.GetImageFromArray(im_moving)
    
    hm = sitk.HistogramMatchingImageFilter()
    hm.SetNumberOfHistogramLevels(levels)
    hm.SetNumberOfMatchPoints(points)

    if mean_intensity:
        hm.ThresholdAtMeanIntensityOn()


    if verbose:
        print('image intensity normalization')
        tic = time.time()
    
    sim_out = hm.Execute(sim1, sim0)
    
    if verbose:
        toc = time.time()
        print('scaling done, {:.3} s'.format(toc - tic))

    im_out = sitk.GetArrayFromImage(sim_out)

    return im_out


def register_im(im_fixed, im_moving, param_map=None, verbose=True):
    '''
    Image registration using SimpleElastix.
    Register im_moving to im_fixed
    '''
    
    sim0 = sitk.GetImageFromArray(im_fixed)
    sim1 = sitk.GetImageFromArray(im_moving)
    
    if param_map is None:
        if verbose:
            print("using default 'transform' parameter map")
        param_map = sitk.GetDefaultParameterMap('translation')

    ef = sitk.ElastixImageFilter()
    ef.SetFixedImage(sim0)
    ef.SetMovingImage(sim1)
    ef.SetParameterMap(param_map)

    if verbose:
        print('image registration')
        tic = time.time()
    
    ef.Execute()
    
    if verbose:
        toc = time.time()
        print('registration done, {:.3} s'.format(toc - tic))

    sim_out = ef.GetResultImage()
    param_map_out = ef.GetTransformParameterMap()

    im_out = sitk.GetArrayFromImage(sim_out)

    return im_out, param_map_out
