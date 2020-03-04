'''
io.py

Input/output loading for contrast synthesis

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/18
'''

import sys
import os # FIXME: transition from os to pathlib
import pathlib
import pydicom
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm

def write_dicoms(input_dicom_folder, output, output_dicom_folder,row=0, col=0,
        series_desc_pre='SubtleGad:', series_desc_post='', series_num=None):
    """Write output numpy array to dicoms, given input dicoms.
    Args:
        input_dicom_folder (str): input dicom folder path.
        output (numpy array): output numpy array.
        output_dicom_folder (str): output dicom folder path.
        custom_series_desc (str): custom series description string.
        Modified from Long Wang
    """
    output_dicom_folder = pathlib.Path(output_dicom_folder)
    output_dicom_folder.mkdir(parents=True, exist_ok=True)

    in_hdrs, in_files, in_names = dicom_header(input_dicom_folder)
    in_data, _  = dicom_files(input_dicom_folder)

    output = np.squeeze(output)
    output_shape = output.shape

    slice_start = in_hdrs[0].SliceLocation
    delta_slice = (in_hdrs[-1].SliceLocation - in_hdrs[0].SliceLocation) / (len(output) - 1)

    x_start = in_hdrs[0].ImagePositionPatient[0]
    delta_x = (in_hdrs[-1].ImagePositionPatient[0] - in_hdrs[0].ImagePositionPatient[0]) / (len(output) - 1)

    y_start = in_hdrs[0].ImagePositionPatient[1]
    delta_y = (in_hdrs[-1].ImagePositionPatient[1] - in_hdrs[0].ImagePositionPatient[1]) / (len(output) - 1)

    z_start = in_hdrs[0].ImagePositionPatient[2]
    delta_z = (in_hdrs[-1].ImagePositionPatient[2] - in_hdrs[0].ImagePositionPatient[2]) / (len(output) - 1)

    dicom = in_hdrs[0]

    dtype = dicom.pixel_array.dtype
    if row == 0 or col == 0:
        row, col = dicom.pixel_array.shape

    dicom.SOPInstanceUID = pydicom.uid.generate_uid()
    dicom.SeriesInstanceUID = pydicom.uid.generate_uid()
    if series_num is None:
        dicom.SeriesNumber = str(int(dicom.SeriesNumber) + 100)
    else:
        dicom.SeriesNumber = str(int(series_num))
    dicom.SliceThickness = abs(delta_z)
    try:
        dicom.StudyDescription = 'SubtleGad:' + dicom.StudyDescription
    except AttributeError:
        pass

    try:
        dicom.SeriesDescription = '{} {} {}'.format(series_desc_pre, dicom.SeriesDescription, series_desc_post)
    except AttributeError:
        pass

    output_min = np.min(output)
    if output_min < 0:
        output = np.copy(output)
        output[np.where(output<0)] = 0
    output = output.astype(dtype)

    for i in tqdm(range(output_shape[0])):
        pixel_array = output[i]
        dicom.InstanceNumber = str(i + 1)
        dicom.SOPInstanceUID = pydicom.uid.generate_uid()
        dicom.ImageIndex = len(output) - i + 1
        dicom.SliceLocation = str(slice_start + delta_slice * i)
        dicom.ImagePositionPatient[0] = str(x_start + delta_x * i)
        dicom.ImagePositionPatient[1] = str(y_start + delta_y * i)
        dicom.ImagePositionPatient[2] = str(z_start + delta_z * i)
        dicom.PixelData = pixel_array.tostring()
        dicom.Rows = row
        dicom.Columns = col
        dicom.save_as(str(output_dicom_folder / '{:03d}.dcm'.format(i)))

def get_dicom_dirs(base_dir, override=False):
    ''' Get list of 'pre', 'low' and 'full' contrast dicom dirs
    For a given base directory, get the subdirectories that
    match the zero/low/post contrast series
    Modified from Akshay's Subtle_SuperRes
    Parameters:
    -----------
    base_dir : string
        name of directory to crawl through
    Returns:
    --------
    dir_list : list
        list of dicom dirs in order of (zero, low, full)
    '''

    # get list of directories
    for fname in glob('{}/**/*.dcm'.format(base_dir), recursive=True):
        base_dir = '/'.join(fname.split('/')[:-2])
        break

    dirs = os.listdir(base_dir)
    dirs_split = np.array([d.split('_') for d in dirs])
    try:
        dirs_sorted = np.array(dirs)[np.argsort([int(dd[0]) for dd in dirs_split])]
    except:
        dirs_sorted = np.array(dirs)[np.argsort([int(dd[-1]) for dd in dirs_split])]

    dir_list = []

    for d in dirs_sorted:
        if override or 'ax' in d.lower() or 'sag' in d.lower():
            if override or 'mprage' in d.lower() or 'bravo' in d.lower():
                if override or 'reformat' not in d.lower():
                    dir_list.append(os.path.join(base_dir, d))
        if len(dir_list) == 3:
            break

    #dir_dict = {
            #'pre': dir_list[0],
            #'low': dir_list[1],
            #'full': dir_list[2].
            #}

    return dir_list


def dicom_files(dicom_dir, normalize=False):

    ''' Load dicom files in a given folder
    For a given dicom directory, load the dicom files that are
    housed in the folder.
    Modified from Akshay's Subtle_SuperRes
    Parameters:
    -----------
    dicom_dir : string
        name of directory to crawl through
    normalize : bool
        whether or not to normalize between 0 and 1
    Returns:
    --------
    img_array : int16/float
        3D array of all dicom files (float if normalized)
    header : pydicom dataset
        header of the dicom scans

    '''

    # build the file list for Dicom images
    lstDCM = []
    for dirName, subdirList, fileList in os.walk(dicom_dir):
        for filename in fileList:
            if ".dcm" in filename.lower() or "mag" in filename.lower():
                lstDCM.append(os.path.join(dirName,filename))

    # sort the list
    lstDCM.sort()

    # Preallocation information

    # get the reference file from the first Dicom image
    hdr = pydicom.read_file(lstDCM[0])
    # matrix dimensions
    nx = int(hdr.Rows)
    ny = int(hdr.Columns)
    nz = len(lstDCM)

    img_array = np.zeros([nz,nx,ny], dtype=hdr.pixel_array.dtype)

    #%% loop through the Dicom list

    for filename in lstDCM:
        try:
            ds = pydicom.read_file(filename)
            img_array[lstDCM.index(filename), :, :] = np.array(ds.pixel_array)
        except Exception:
            continue
    #%% Normalize the array

    if normalize is True:
        img_array = img_array/np.amax(img_array)

    return np.float32(img_array), hdr

#%% Load dicom files function

def dicom_header(dicom_dir):

    ''' Load dicom header from dicom files in a given folder
    For a given dicom directory, load the dicom header that are
    housed in the folder. Same function as load_dicom_files
    but only includes header
    Parameters:
    -----------
    dicom_dir : string
        name of directory to crawl through
    Returns:
    --------
    header : pydicom dataset
        header of the dicom scans
    lstDcm : pydicom list dataset
        list of dicom files in folder
    fileOut : list
        list of full locations of all dicoms

    '''

    # build the file list for Dicom images
    lstDCM = []
    fileOut = []
    for dirName, subdirList, fileList in os.walk(dicom_dir):
        for filename in fileList:
            if ".dcm" in filename.lower() or 'mag' in filename.lower():
                lstDCM.append(os.path.join(dirName,filename))
                fileOut.append(filename)


    # sort the list
    lstDCM.sort()
    fileOut.sort()

    # Preallocation information

    # get the reference file from the first Dicom image
    hdr_list = [pydicom.read_file(h) for h in lstDCM]

    return hdr_list, lstDCM, fileOut


def get_pixel_spacing(hdr):
    ''' Get pixel spacing from dicom header '''
    x_spacing, y_spacing = hdr.PixelSpacing
    z_spacing = hdr.SliceThickness
    return x_spacing, y_spacing, z_spacing


def get_npy_files(data_dir, max_data_sets=np.inf):
    ''' Get list of npy files in a given directory

    Parameters:
    -----------
    data_dir : string
        name of directory to crawl through
    max_data_sets : int
        maximum number of data sets to load

    Returns:
    --------
    npy_list : list
        list of npy files
    '''
    # build the file list for npy files
    npy_list = []
    for dir_name, subdir_list, file_list in os.walk(data_dir):
        for filename in file_list:
            if '.npy' in filename.lower() and len(npy_list) < max_data_sets:
                npy_list.append(os.path.join(dir_name, filename))

    # sort the list
    npy_list.sort()

    return npy_list

def load_file(input_file, file_type=None, params={'h5_key': 'data'}):
    return load_slices(input_file, slices=None, file_type=file_type, params=params)

def load_h5_file(h5_file, h5_key='data'):
    return load_slices_h5(h5_file, slices=None, h5_key=h5_key)

def load_h5_metadata(h5_file, key='metadata'):
    metadata = {}
    with h5py.File(h5_file, 'r') as F:
        for k in F[key].keys():
            metadata[k] = np.array(F[key][k])
    return metadata


def load_npy_file(npy_file):
    return load_slices_npy(npy_file, slices=None)

def load_npy_files(data_dir, npy_list=None, max_data_sets=np.inf):

    ''' Load npy files in a given directory
    For a given directory, load the npy files that are
    housed in the folder and return as a list of numpy arrays

    Parameters:
    -----------
    data_dir : string
        name of directory to crawl through
    npy_list : list
        list of npy files. If None, then compute it
    max_data_sets : int
        maximum number of data sets to load

    Returns:
    --------
    out : list
        list containing all the np arrays
    '''

    if npy_list is None:
        npy_list = get_npy_files(data_dir, max_data_sets=max_data_sets)

    out = []

    # loop through the npy list

    for filename in npy_list:
        # transpose into format expected by Keras
        # [ns, nx, ny, 3]
        out.append(load_npy_file(filename))

    return out

def get_data_list(data_list_file, data_dir=None, file_ext=None):
    f = open(data_list_file, 'r')
    data_list = []
    for l in f.readlines():
        s = l.strip()
        if data_dir is not None:
            s = '{}/{}'.format(data_dir, s)
        if file_ext is not None:
            s = '{}.{}'.format(s, file_ext)
        data_list.append(s)
    f.close()
    return data_list

def get_file_type(input_file):
    ''' Get file type from input file

    Parameters:
    -----------
    input_file : string
        name of data file

    Returns:
    --------
    file_type : str
        string representing file type
    '''

    suffix = ''.join(pathlib.Path(input_file).suffixes)

    if suffix in ['.npy', '.npz']:
        return 'npy'
    elif suffix in ['.h5', '.hdf5', '.h5z']:
        return 'h5'
    else:
        return -1


def save_data_npy(output_file, data):
    try:
        np.save(output_file, data)
        return 0
    except:
        return -1

def save_data_h5(output_file, data, data_mask=None, h5_key='data', compress=False, metadata=None):
    with h5py.File(output_file, 'w') as f:
        h5_params = {}

        if compress:
            h5_params['compression'] = 'gzip'

        h5_params['data'] = data
        f.create_dataset(h5_key, **h5_params)

        if data_mask is not None:
            h5_params['data'] = data_mask
            f.create_dataset('data_mask', **h5_params)

        f.close()

    return 0

def save_meta_h5(output_file, metadata):
    h5_params = {}
    with h5py.File(output_file, 'w') as f:
        for key in metadata.keys():
            _h5_key = 'metadata/{}'.format(key)
            metakey = metadata[key]
            if callable(metakey):
                metakey = metakey.__name__
            h5_params['data'] = metakey
            f.create_dataset(_h5_key, **h5_params)
        f.close()
    return 0

def save_data(output_file, data, file_type=None, params={'h5_key': 'data', 'compress': False}):
    ''' Save data to output file using file type format

    Parameters:
    -----------
    output_file : string
        name of output data file
    data : numpy array
        numpy array to save
    file_type : string
        defines the file type of the input data
    params : dict
        dictionary used for loading the data

    Returns:
    --------
    code : int
        exit code (0 success, non-zero fail)
    '''

    if file_type is None:
        file_type = get_file_type(output_file)

    if file_type == 'h5':
        return save_data_h5(output_file, data, h5_key=params['h5_key'], compress=params['compress'])
    else:
        # default to npy
        return save_data_npy(output_file, data)

def get_shape(input_file, file_type=None, params={'h5_key': 'data'}):
    ''' Get shape of data from data file

    Parameters:
    -----------
    input_file : string
        name of data file
    file_type : string
        defines the file type of the input data
    params : dict
        dictionary used for loading the data

    Returns:
    --------
    data_shape : numpy array
        numpy array of dimensions
    '''

    if file_type is None:
        file_type = get_file_type(input_file)

    if file_type == 'npy':
        return get_shape_npy(input_file)
    elif file_type == 'h5':
        return get_shape_h5(input_file, h5_key=params['h5_key'])
    else:
        print('subtle_io/get_shape: ERROR. unrecognized file type', file_type)
        sys.exit(-1)

def get_shape_npy(input_file):
    f = np.load(input_file, mmap_mode='r')
    fs = f.shape
    if len(fs) > 4:
        return fs[-4:]
    return fs

def get_shape_h5(input_file, h5_key='data'):
    F = h5py.File(input_file, 'r')
    data_shape = F[h5_key].shape
    F.close()
    return data_shape

def has_h5_key(fpath_h5, key):
    h5_file = h5py.File(fpath_h5, 'r')
    has_key = key in h5_file
    h5_file.close()

    return has_key

def load_slices_h5(input_file, slices=None, h5_key='data', dim=0):
    # FIXME: remove code duplication
    F = h5py.File(input_file, 'r')
    if slices is None: # load the full volume
        if h5_key == 'all':
            d1 = np.array(F['data'])
            d2 = np.array(F['data_mask'])
            data = np.array([d1, d2])
        else:
            data = np.array(F[h5_key])
    else:
        slices_unique, slices_inverse = np.unique(slices, return_index=False, return_inverse=True, return_counts=False)
        if dim == 0:
            if h5_key == 'all':
                d1 = np.array(F['data'][slices_unique, :, :, :])
                d2 = np.array(F['data_mask'][slices_unique, :, :, :])
                if len(slices_unique) < len(slices_inverse):
                    d1 = d1[slices_inverse, :, :, :]
                    d2 = d2[slices_inverse, :, :, :]
                data = np.array([d1, d2])
            else:
                data = np.array(F[h5_key][slices_unique, :, :, :])
                if len(slices_unique) < len(slices_inverse):
                    data = data[slices_inverse, :, :, :]
        elif dim == 1:
            if h5_key == 'all':
                d1 = np.array(F['data'][:, slices_unique,  :, :])
                d2 = np.array(F['data_mask'][:, slices_unique,  :, :])
                if len(slices_unique) < len(slices_inverse):
                    d1 = d1[:, slices_inverse, :, :]
                    d2 = d2[:, slices_inverse, :, :]
                data = np.array([d1, d2])
            else:
                data = np.array(F[h5_key][:, slices_unique,  :, :])
                if len(slices_unique) < len(slices_inverse):
                    data = data[:, slices_inverse, :, :]
        elif dim == 2:
            if h5_key == 'all':
                d1 = np.array(F['data'][:, :, slices_unique, :])
                d2 = np.array(F['data_mask'][:, :, slices_unique, :])
                if len(slices_unique) < len(slices_inverse):
                    d1 = d1[:, :, slices_inverse, :]
                    d2 = d2[:, :, slices_inverse, :]
                data = np.array([d1, d2])
            else:
                data = np.array(F[h5_key][:, :, slices_unique, :])
                if len(slices_unique) < len(slices_inverse):
                    data = data[:, :, slices_inverse, :]
        elif dim == 3:
            if h5_key == 'all':
                d1 = np.array(F['data'][:, :, :, slices_unique])
                d2 = np.array(F['data_mask'][:, :, :, slices_unique])
                if len(slices_unique) < len(slices_inverse):
                    d1 = d1[:, :, :, slices_inverse]
                    d2 = d2[:, :, :, slices_inverse]
                data = np.array([d1, d2])
            else:
                data = np.array(F[h5_key][:, :, :, slices_unique])
                if len(slices_unique) < len(slices_inverse):
                    data = data[:, :, :, slices_inverse]
    F.close()
    return data

def load_slices_npy(input_file, slices=None, dim=0):
    d = np.load(input_file, mmap_mode='r')
    if slices is None:
        return d
    else:
        if dim == 0:
            return d[slices, :, :, :]
        elif dim == 1:
            return d[:, slices, :, :]
        elif dim == 2:
            return d[:, :, slices, :]
        elif dim == 3:
            return d[:, :, :, slices]

def load_slices(input_file, slices=None, file_type=None, params={'h5_key': 'data'}, dim=0):
    ''' Load some or all slices from data file

    Parameters:
    -----------
    input_file : string
        name of data file
    slices : numpy array or list
        list of slices to load. If None, load all slices
    file_type : string
        defines the file type of the input data
    params : dict
        dictionary used for loading the data

    Returns:
    --------
    out : numpy array
        numpy array
    '''

    if file_type is None:
        file_type = get_file_type(input_file)

    if file_type == 'npy':
        return load_slices_npy(input_file, slices, dim=dim)
    elif file_type == 'h5':
        return load_slices_h5(input_file, slices, h5_key=params['h5_key'], dim=dim)
    else:
        #FIXME add real exception handling
        print('subtle_io/load_slices: ERROR. unrecognized file type', file_type)
        sys.exit(-1)