'''
subtle_metrics.py

Metrics for contrast synthesis

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/09/25
'''

import sys
import warnings
import time
import numpy as np

try:
    from skimage.measure import compare_ssim as ssim_score
except:
    from skimage.metrics import structural_similarity as ssim_score

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def nrmse(x_truth, x_predict, axis=None):
    ''' Calculate normalized root mean squared error
    along a given axis. NRMSE is defined as
    norm(x_truth - x_predict, 2) / norm(x_truth, 2)

    Parameters:
    -----------
    x_truth : numpy ndarray
        original data
    x_predict : numpy ndarray
        comparison data
    axis : int
        norm axis

    Returns:
    -----------
    nrmse : float or numpy ndarray
        resulting nrmse along axis
    '''

    return np.linalg.norm(x_truth - x_predict, axis=axis) / np.linalg.norm(x_truth, axis=axis)

def normalize_ims(x_truth, x_predict):
    ''' Normalize images w.r.t. x_truth
    x_truth and x_predict are normalized to either (-.5, .5) if negative
    or to (0, 1) if positive

    Parameters:
    -----------
    x_truth : numpy ndarray
        original data
    x_predict : numpy ndarray
        comparison data

    Returns:
    -----------
    x_truth_nrm : float or numpy ndarray
        x_truth after normalization
    x_predict_nrm : float or numpy ndarray
        x_predict after normalization
    '''
    if np.all(x_truth >= 0):
        max_val = np.max(x_truth)
        x_truth_nrm = x_truth / max_val
        x_predict_nrm = x_predict / max_val
    else:
        max_val = np.max(abs(x_truth))
        x_truth_nrm = x_truth / max_val / 2.
        x_predict_nrm = x_predict / max_val / 2.
    return x_truth_nrm, x_predict_nrm

def psnr(x_truth, x_predict, axis=None, dynamic_range=None):
    ''' Calculate PSNR along a given axis.
    PSNR is defined as
    10 log10(dynamic_range**2 / MSE(x_truth - x_predict)),
    where dynamic_range is the dynamic range (e.g. 255 for uint8 images)
    and MSE is the mean-squared error.
    If dynamic_range is None, then x_truth and x_predict
    are normalized to either (-.5, .5) if negative or to (0, 1) if positive
    and dynamic_range is set to 1.

    Parameters:
    -----------
    x_truth : numpy ndarray
        original data
    x_predict : numpy ndarray
        comparison data
    axis : int
        norm axis
    dynamic_range : float
        dynamic range

    Returns:
    -----------
    psnr : float or numpy ndarray
        resulting psnr along axis
    '''

    if dynamic_range is None:
        x_truth, x_predict = normalize_ims(x_truth, x_predict)
        dynamic_range = 1.

    if axis is None:
        nrm = len(x_truth.ravel())
    else:
        nrm = x_truth.shape[axis]

    MSE = np.linalg.norm(x_truth - x_predict, axis=axis)**2 / nrm

    return 20 * np.log10(dynamic_range) - 10 * np.log10(MSE)


def ssim(x_truth, x_predict, axis=None, dynamic_range=None):

    ''' Calculate SSIM along a given axis.
    SSIM depends on the dynamic range (e.g. 255 for uint8 images).
    If dynamic_range is None, then x_truth and x_predict
    are normalized to either (-.5, .5) if negative or to (0, 1) if positive
    and dynamic_range is set to 1.

    Parameters:
    -----------
    x_truth : numpy ndarray
        original data
    x_predict : numpy ndarray
        comparison data
    axis : int
        norm axis
    dynamic_range : float
        dynamic range

    Returns:
    -----------
    psnr : float or numpy ndarray
        resulting psnr along axis
    '''

    if dynamic_range is None:
        x_truth, x_predict = normalize_ims(x_truth, x_predict)
        dynamic_range = 1.

    if axis is None:
        nrm = len(x_truth.ravel())
    else:
        nrm = x_truth.shape[axis]

    if x_truth.dtype != x_predict.dtype:
        warnings.warn('x_truth.dtype == {} != {} == x_predict.dtype. Casting x_predict to x_truth'.format(x_truth.dtype, x_predict.dtype))
        x_predict = x_predict.astype(dtype=x_truth.dtype)

    score = ssim_score(x_truth, x_predict, data_range=dynamic_range, channel_axis=0)
    return score
