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
from skimage.morphology import binary_dilation, binary_erosion, rectangle, square
from scipy.ndimage.interpolation import zoom as zoom_interp
from scipy.ndimage import label as cc_label
import cv2
from dicom2nifti.convert_dicom import dicom_series_to_nifti
import nibabel as nib
import pydicom
from skimage.measure import label, regionprops
from glob import glob

try:
    sys.path.insert(0, '/home/subtle/jon/tools/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.5/SimpleITK')
    import SimpleITK as sitk
except:
    warnings.warn('SimpleITK not found!')


def scale_slope_intercept(im, rs, ri, ss):
    return (im * rs + ri) / (rs * ss)
def rescale_slope_intercept(im, rs, ri, ss):
    return (im * rs * ss - ri) / rs


# FIXME: do differently for each image
def mask_im(im, threshold=.08, noise_mask_area=False):
    '''
    Image masking
    Masks an image set based on max val compared to threshold.
    Fills holes in main mask
    '''
    N, n, nx, ny = im.shape
    mask = im > (threshold * np.amax(im, axis=(2,3))[:,:,None,None])
    # fill holes in mask
    mask = binary_fill_holes(mask.reshape((n*N*nx, ny))).reshape((N, n, nx, ny))

    if noise_mask_area:
        mask = binary_erosion(mask.reshape((n*N*nx, ny)), selem=rectangle(7, 4)).reshape((N, n, nx, ny))

        for cont in np.arange(n):
            mask[:, cont, ...] = get_largest_connected_component(mask[:, cont, ...])

    return mask

def get_largest_connected_component(mask):
    components = cc_label(mask)[0]
    max_area = np.max([reg.area for reg in regionprops(components)])

    for val in np.unique(components):
        if val != 0:
            mask_label = (components == val)
            if np.sum(mask_label) == max_area:
                return mask_label

def normalize_data(data, verbose=False, fun=np.mean, axis=(0,1,2), nslices=5):
    ntic = time.time()
    if verbose:
        print('normalizing data')
    data_out = normalize_im(data.copy(), axis=axis, fun=fun) # normalize each contrast separately by default
    ntoc = time.time()
    if verbose:
        print('done ({:.2f}s)'.format(ntoc - ntic))

    if verbose:
        print('scaling data')
    ntic = time.time()
    nz = data_out.shape[0]
    idx_scale = range(nz//2 - nslices, nz//2 + nslices)
    scale_low = scale_im_enhao(data_out[idx_scale, :, :, 0], data_out[idx_scale, :, :, 1])
    scale_full = scale_im_enhao(data_out[idx_scale, :, :, 0], data_out[idx_scale, :, :, 2])
    ntoc = time.time()
    if verbose:
        print('scale low:', scale_low)
        print('scale full:', scale_full)
        print('done scaling data ({:.2f} s)'.format(ntoc - ntic))
    data_out[:,:,:,1] = data_out[:,:,:,1] * scale_low
    data_out[:,:,:,2] = data_out[:,:,:,2] * scale_full

    return data_out

def normalize_im(im, axis=None, fun=np.mean):
    '''
    Image normalization
    Normalizes an image along the axis dimensions using the function fun
    '''

    sc = normalize_scale(im, axis, fun)
    return im / sc

def normalize_scale(im, axis=None, fun=np.mean):
    '''
    Image normalization
    Returns the normalization of an image along the axis dimensions using the function fun
    '''
    im[im < 0] = 0

    if type(axis) == int:
        axis = (axis,)

    if axis is None:
        sc = fun(im.ravel())
    else:
        sc = fun(im, axis=axis)
        for i in axis:
            sc = np.expand_dims(sc, axis=i)
    return sc


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
    sim0 = sitk.GetImageFromArray(im_fixed.squeeze())
    sim1 = sitk.GetImageFromArray(im_moving.squeeze())

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
    im_out = np.clip(im_out, 0, im_out.max())

    return im_out, param_map_out

def apply_reg_transform(img, spacing, transform_params):
    transform_filter = sitk.TransformixImageFilter()

    simg = sitk.GetImageFromArray(img)
    simg.SetSpacing(spacing)

    params = transform_params[0]

    simg_trans = sitk.Transformix(simg, params)
    simg_arr = sitk.GetArrayFromImage(simg_trans)
    return simg_arr

def undo_scaling(im_predict, metadata, verbose=False, im_gt=None):
    ''' Applies the inverse of the scaling/normalization from preprocessing.
    Inputs:
    im_predict (ndarray): input volume
    metadata (dict): dictionary containing scaling metadata
    Outputs:
    out (ndarray): volume after inverse of scaling
    '''

    out = im_predict.copy()
    key1 = 'scale_global'

    if key1 in  metadata.keys():
        if 'global_scale_ref_im0' in metadata and metadata['global_scale_ref_im0']:
            sc = metadata[key1][0]
        else:
            sc = metadata[key1][0][0]
        if verbose:
            print('re-scaling by global scale', sc)
        # using x0 scale because we will not have access to x2 at inference time
        out = out * sc

    key = 'dicom_scaling_zero'
    if key in metadata.keys():
        if verbose:
            print('re-scaling by dicom tags{}'.format(key), metadata[key])
        rs, ri, ss = metadata[key]
        out = rescale_slope_intercept(out, rs, ri, ss)


    # FIXME: is this necessary? data are already scaled to match pre-con
    key2 = 'scale_zero'
    if key2 in metadata.keys():
        if verbose:
            print('re-scaling by {}'.format(key2), metadata[key2])
        out = out / metadata[key2]


    # undo histogram normalization. as a quick test I am using x2 for this, but in the future we should only be using a template image

    # levels=1024
    # points=50
    # mean_intensity=True
    # key3 = 'hist_norm'
    # if key3 in metadata.keys():
    #     if verbose:
    #         print('undoing histogram normalization (TODO: use template)'.format(key3))
    #         out = scale_im(im_gt, out[..., 0].astype(im_gt.dtype), levels, points, mean_intensity)
    #         out = np.array([out]).transpose(1, 2, 3, 0)

    return out

def resample_slices(slices, resample_size=None):
    if resample_size is None or resample_size == slices.shape[2]:
        return slices

    num_slices, num_cont, _, _ = slices.shape
    slices_resample = np.zeros((num_slices, num_cont, resample_size, resample_size), dtype=slices.dtype)

    for slice_num in range(num_slices):
        for cont_num in range(num_cont):
            slices_resample[slice_num, cont_num, ...] = cv2.resize(slices[slice_num, cont_num], dsize=(resample_size, resample_size), interpolation=cv2.INTER_CUBIC)

    return slices_resample

def dcm2nii(dcmdir, out_dir):
    out_file = '{}/{}.nii'.format(out_dir, dcmdir.split('/')[-1])
    dicom_series_to_nifti(dcmdir, out_file, reorient_nifti=False)
    return out_file

def nii2npy(fpath_nii, transpose=True):
    nii_img = nib.load(fpath_nii)
    img = nii_img.get_fdata()

    if transpose:
        img = np.transpose(img, (2, 1, 0))

    return img

def get_brain_area_cm2(mask, spacing=[1., 1., 1.]):
    spacing_x, spacing_y = np.array(spacing[1:], dtype=np.float32)
    binary_mask = np.copy(mask).astype(np.int8)

    props = regionprops(binary_mask)
    if not props:
        return 0.0

    area_cm2 = (props[0].area * spacing_x * spacing_y) / 1e3
    return area_cm2

def get_brain_portion(img, threshold=0):
    center_mask = binary_fill_holes(img[img.shape[0] // 2] > threshold)

    regions = regionprops(label(center_mask))
    if len(regions) == 1:
        bbox = regions[0].bbox
    else:
        sorted_regions = sorted(regions, key=lambda r: r.bbox_area)
        bbox = sorted_regions[-1].bbox

    (x1, y1, x2, y2) = bbox
    cropped_img = img[:, x1:x2, y1:y2]

    return cropped_img, bbox

def _get_crop_range(shape_num):
    if shape_num % 2 == 0:
        start = end = shape_num // 2
    else:
        start = (shape_num + 1) // 2
        end = (shape_num - 1) // 2

    return (start, end)

def center_position_brain(img, threshold=0):
    cropped_img, bbox = get_brain_portion(img, threshold)

    x_start, x_end = _get_crop_range(img.shape[1] - cropped_img.shape[1])
    y_start, y_end = _get_crop_range(img.shape[2] - cropped_img.shape[2])

    new_img = np.zeros_like(img)
    new_img[:, x_start:-x_end, y_start:-y_end] = cropped_img

    return new_img, bbox

def center_crop(img, ref_img):
    s = []
    e = []

    for i, sh in enumerate(img.shape):
        if sh > ref_img.shape[i]:
            diff = sh - ref_img.shape[i]
            if diff == 1:
                s.append(0)
                e.append(sh-1)
            else:
                start, end = _get_crop_range(diff)
                s.append(start)
                e.append(-end)
        else:
            s.append(0)
            e.append(sh)

    new_img = img[s[0]:e[0], s[1]:e[1], s[2]:e[2]]

    if new_img.shape[1] != new_img.shape[2]:
        new_img = zoom_interp(new_img, [1, ref_img.shape[1]/new_img.shape[1], ref_img.shape[2]/new_img.shape[2]])

    return new_img

def zero_pad_for_dnn(img, num_poolings=3):
    mod_check = (2 ** num_poolings)
    mod_filter = [(s % mod_check != 0) for s in img.shape]

    if not np.any(mod_filter):
        return img

    npad = []
    for i, sh in enumerate(img.shape):
        if sh % mod_check == 0 or i == 1:
            npad.append((0, 0))
        else:
            new_shape = sh + (sh % mod_check)
            if new_shape % mod_check != 0:
                new_shape = sh + (mod_check - (sh % mod_check))
            npad.append(_get_crop_range(new_shape - sh))
    return np.pad(img, pad_width=npad, mode='constant', constant_values=0)

def zero_pad(img, target_size=256):
    npad = []

    if img.ndim == 4:
        diff1 = (target_size - img.shape[2]) // 2
        diff2 = (target_size - img.shape[3]) // 2
        npad = [(0, 0), (0, 0), (diff1, diff1), (diff2, diff2)]
    elif img.ndim == 3:
        diff1 = (target_size - img.shape[1]) // 2
        diff2 = (target_size - img.shape[2]) // 2
        npad = [(0, 0), (diff1, diff1), (diff2, diff2)]
    else:
        return img

    return np.pad(img, pad_width=npad, mode='constant', constant_values=0)

def undo_brain_center(img, bbox, threshold=0):
    cropped_img, _ = get_brain_portion(img, threshold)

    (x1, y1, x2, y2) = bbox
    _, x_shape, y_shape = cropped_img.shape

    new_img = np.zeros_like(img)

    new_img[:, x1:x1+cropped_img.shape[1], y1:y1+cropped_img.shape[2]] = cropped_img

    return new_img

def zoom_iso(dcm_vol, spacing, new_spacing):
    resize_factor = spacing / new_spacing
    new_shape = np.round(dcm_vol.shape * resize_factor)

    real_resize_factor = new_shape / dcm_vol.shape
    new_spacing = spacing / real_resize_factor

    return zoom_interp(dcm_vol, real_resize_factor), new_spacing

def enhancement_mask(X, Y, center_slice, th=.05):
    'Create an enhancement mask to weight the loss function'

    # FIXME: only works if Y.shape[2] == 1
    im_diff = Y[:, 0, 0, ...].squeeze() - X[:, center_slice, 0,...].squeeze()
    enh_mask = im_diff > th * np.max(abs(im_diff))
    enh_mask[enh_mask < 0] = 0
    if len(enh_mask.shape) > 2:
        for i in range(enh_mask.shape[0]):
            enh_mask[i,...] = binary_dilation(enh_mask[i, ...], selem=square(3))
    else:
        enh_mask = binary_dilation(enh_mask, selem=square(3))

    enh_mask = enh_mask.reshape(Y.shape)
    return enh_mask
