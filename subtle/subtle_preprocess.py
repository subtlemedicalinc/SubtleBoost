'''
subtle_preprocess.py

Pre-processing utilities for contrast synthesis
Mask, intensity normalization, and image co-registration

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/18
'''

import sys
import warnings
import time

import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

try:
    sys.path.insert(0, '/home/subtle/jon/tools/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.5/SimpleITK')
    import SimpleITK as sitk
except:
    warnings.warn('SimpleITK not found!')
    
    
def mask_im(im, threshold=.08):
    '''
    Image masking
    Masks an image set based on max val compared to threshold.
    Fills holes in main mask
    '''
    N, n, nx, ny = im.shape
    mask = im > (threshold * np.amax(im, axis=(2,3))[:,:,None,None])
    mask = np.amax(mask, axis=1)[:,None,:,:].repeat(axis=1, repeats=n)
    # fill holes in mask
    mask = binary_fill_holes(mask.reshape((n*N*nx, ny))).reshape((N, n, nx, ny))
    return mask   

def normalize_data(data, verbose=False, fun=np.mean):
    ntic = time.time()
    if verbose:
        print('normalizing data')
    data_out = normalize_im(data.copy(), axis=(0,1,2), fun=fun) # normalize each contrast separately
    ntoc = time.time()
    if verbose:
        print('done ({:.2f}s)'.format(ntoc - ntic))

    if verbose:
        print('scaling data')
    ntic = time.time()
    nz = data_out.shape[0]
    idx_scale = range(nz//2 - 5, nz//2 + 5)
    scale_low = scale_im_enhao(data_out[idx_scale, :, :, 0], data_out[idx_scale, :, :, 1])
    scale_full = scale_im_enhao(data_out[idx_scale, :, :, 0], data_out[idx_scale, :, :, 2])
    ntoc = time.time()
    if verbose:
        print('scale low:', scale_low)
        print('scale full:', scale_full)
        print('done scaling data ({:.2f} s)'.format(ntoc - ntic))
    data_out[:,:,:,1] = data_out[:,:,:,1] / scale_low
    data_out[:,:,:,2] = data_out[:,:,:,2] / scale_full

    return data_out

def normalize_im(im, axis=None, fun=np.mean):
    '''
    Image normalization
    Normalizes an image along the axis dimensions using the function fun
    '''
    im[im < 0] = 0

    if type(axis) == int:
        axis = (axis,)

    if axis is None:
        return im / fun(im.ravel())
    else:
        sc = fun(im, axis=axis)
        for i in axis:
            sc = np.expand_dims(sc, axis=i)
        return im / sc


def scale_im_enhao(im_fixed, im_moving, levels=np.linspace(.8,1.2,30), fun=lambda x: np.mean(np.abs(x[np.abs(x)>0.1].ravel())), max_iter=1):
    '''
    Image intensity scaling based on Enhao's approach
    Returns the scale factor that adjusts im_moving to the scale of im_fixed according to fun
    '''
    best_scale = levels.mean()
    best_cost = np.inf

    for index_iter in range(max_iter):
        for level in levels:
            im_moving_sc = im_moving * level
            im_diff = im_moving_sc - im_fixed
            diff_cost = fun(im_diff)
            if diff_cost < best_cost:
                best_scale = level
                best_cost = diff_cost
        delta_scale = levels[1] - levels[0]
        levels = np.linspace(best_scale - delta_scale, best_scale + delta_scale, len(levels))

    return best_scale  


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


def register_im(im_fixed, im_moving, param_map=None, verbose=True, im_fixed_spacing=None, im_moving_spacing=None):
    '''
    Image registration using SimpleElastix.
    Register im_moving to im_fixed
    '''

    default_transform = 'translation'
    
    sim0 = sitk.GetImageFromArray(im_fixed)
    sim1 = sitk.GetImageFromArray(im_moving)

    if im_fixed_spacing is not None:
        sim0.SetSpacing(im_fixed_spacing)

    if im_moving_spacing is not None:
        sim1.SetSpacing(im_moving_spacing)
    
    if param_map is None:
        if verbose:
            print("using default '{}' parameter map".format(default_transform))
        param_map = sitk.GetDefaultParameterMap(default_transform)

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
