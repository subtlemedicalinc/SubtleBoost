'''
subtle_io.py

Input/output loading for contrast synthesis
dicom read/write functions

@author: Jon Tamir (jon@subtlemedical.com)
Modified from Akshay's Subtle_SuperRes
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/18
'''

import sys
import os # FIXME: transition from os to pathlib
import pathlib

import h5py

import numpy as np

try:
    import dicom as pydicom
except:
    import pydicom
try:
    import keras
except:
    pass

import subtle.subtle_preprocess as sup


def get_dicom_dirs(base_dir):

    ''' Get list of 'pre', 'low' and 'full' contrast dicom dirs
    For a given base directory, get the subdirectories that
    match the zero/low/post contrast series
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
    dirs = os.listdir(base_dir)
    dirs_split = np.array([d.split('_') for d in dirs])
    dirs_sorted = np.array(dirs)[np.argsort([int(dd[0]) for dd in dirs_split])]

    dir_list = []

    for d in dirs_sorted:
        if 'ax' in d.lower():
            if 'mprage' in d or 'bravo' in d.lower():
                dir_list.append(os.path.join(base_dir, d))
        if len(dir_list) == 3:
            break

    #dir_dict = {
            #'pre': dir_list[0],
            #'low': dir_list[1],
            #'full': dir_list[2].
            #}

    return dir_list


def dicom_files(dicom_dir, normalize = False):
    
    ''' Load dicom files in a given folder
    For a given dicom directory, load the dicom files that are
    housed in the folder.
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
            if ".dcm" in filename.lower():
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
        ds = pydicom.read_file(filename)
        img_array[lstDCM.index(filename), :, :] = np.array(ds.pixel_array)
    
    #%% Normalize the array
    
    if normalize is True:
        img_array = img_array/np.amax(img_array)
        
    return img_array, hdr

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
            if ".dcm" in filename.lower():
                lstDCM.append(os.path.join(dirName,filename))
                fileOut.append(filename)

    
    # sort the list
    lstDCM.sort()
    fileOut.sort()
    
    # Preallocation information
    
    # get the reference file from the first Dicom image
    hdr = dicom.read_file(lstDCM[0])
        
    return hdr, lstDCM, fileOut


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
    return load_slices_h5(h5_file, slices=None, h5_key=k5_key)

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

def save_data_h5(output_file, data, h5_key='data', compress=True):
    try:
        with h5py.File(args.output_file, 'w') as f:
            if compress:
                f.create_dataset(h5_key, data=data, compression='gzip')
            else:
                f.create_dataset(h5_key, data=data)
        return 0
    except:
        return -1

def save_data(output_file, data, file_type=None, params={'h5_key': 'data', 'compress': True}):
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

def load_slices_h5(input_file, slices=None, h5_key='data'):
    F = h5py.File(input_file, 'r')
    if slices is None:
        data = np.array(F[h5_key])
    else:
        slices_unique, slices_inverse = np.unique(slices, return_index=False, return_inverse=True, return_counts=False)
        data = np.array(F[h5_key][slices_unique, :, :, :])
        if len(slices_unique) < len(slices_inverse):
            data = data[slices_inverse, :, :, :]
    F.close()
    return data

def load_slices_npy(input_file, slices=None):
    d = np.load(k, mmap_mode='r')
    if slices is None:
        return d
    else:
        return d[slices, :, :, :]

def load_slices(input_file, slices=None, file_type=None, params={'h5_key': 'data'}):
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
        return load_slices_npy(input_file, slices)
    elif file_type == 'h5':
        return load_slices_h5(input_file, slices, h5_key=params['h5_key'])
    else:
        print('subtle_io/load_slices: ERROR. unrecognized file type', file_type)
        sys.exit(-1)


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
    return f.shape

def get_shape_h5(input_file, h5_key='data'):
    F = h5py.File(input_file, 'r')
    data_shape = F[h5_key].shape
    F.close()
    return data_shape


def get_num_slices(data_file, axis=0, file_type=None, params={'h5_key': 'data'}):
    ''' Get number of slices along a particular axis in data file

    Parameters:
    -----------
    input_file : string
        name of data file
    axis : int
        axis to check for slices
    file_type : string
        defines the file type of the input data
    params : dict
        dictionary used for loading the data

    Returns:
    --------
    num : int
        number of slices in dimension axis
    '''

    data_shape = get_shape(data_file, file_type, params)
    return data_shape[axis]


def build_slice_list(data_list):
    ''' Builds two lists where the index is the
    slice number and the value is the file name / slice index.
    The length of the list is the total number of slices

    Parameters:
    -----------
    data_list : list
        list of file names

    Returns:
    --------
    slice_list_files : list
        slice list corresponding to files
    slice_list_indexes : list
        slice list corresponding to slice indexes
    '''

    slice_list_files = []
    slice_list_indexes = []

    for data_file in data_list:
        num_slices = get_num_slices(data_file)
        slice_list_files.extend([data_file] * num_slices)
        slice_list_indexes.extend(range(num_slices))

    return slice_list_files, slice_list_indexes

# https://stackoverflow.com/questions/15722324/sliding-window-in-numpy
def window_stack(a, stepsize=1, width=3):
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_list, batch_size=8, slices_per_input=1, shuffle=True, verbose=1, residual_mode=True):

        'Initialization'
        self.data_list = data_list
        self.batch_size = batch_size
        self.slices_per_input = slices_per_input # 2.5d
        self.shuffle = shuffle
        self.verbose = verbose
        self.residual_mode = residual_mode

        _slice_list_files, _slice_list_indexes = build_slice_list(self.data_list)
        self.slice_list_files = np.array(_slice_list_files)
        self.slice_list_indexes = np.array(_slice_list_indexes)

        self.slices_per_file_dict = {data_file: get_num_slices(data_file) for data_file in self.data_list}
        self.num_slices = len(self.slice_list_files)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_slices / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        if self.verbose > 1:
            print('batch index:', index)

        file_list = self.slice_list_files[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
        slice_list = self.slice_list_indexes[self.indexes[index*self.batch_size:(index+1)*self.batch_size]]

        if self.verbose > 1:
            print('list of files and slices:')
            print(file_list)
            print(slice_list)

        # Generate data
        X, Y = self.__data_generation(file_list, slice_list)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_slices)
        if self.shuffle == True:
            self.indexes = np.random.permutation(self.indexes)

    def __data_generation(self, slice_list_files, slice_list_indexes):
        'Generates data containing batch_size samples' 

        # load volumes
        # each element of the data_list contains 3 sets of 3D
        # volumes containing zero, low, and full contrast.
        # the number of slices may differ but the image dimensions
        # should be the same

        data_list_X = []
        data_list_Y = []

        for f, c in zip(slice_list_files, slice_list_indexes):
            num_slices = self.slices_per_file_dict[f]
            h = self.slices_per_input // 2

            # 2.5d
            idxs = np.arange(c - h, c + h + 1)

            # handle edge cases for 2.5d by just repeating the boundary slices
            idxs = np.minimum(np.maximum(idxs, 0), num_slices - 1)

            slices = load_slices(input_file=f, slices=idxs) # [c, 3, ny, nz]

            slices_X = slices[:,:2,:,:][None,:,:,:,:]
            slice_Y = slices[h, -1, :, :][None,:,:] 

            data_list_X.append(slices_X)
            data_list_Y.append(slice_Y)
            
        if len(data_list_X) > 1:
            data_X = np.concatenate(data_list_X, axis=0)
            data_Y = np.concatenate(data_list_Y, axis=0)
        else:
            data_X = data_list_X[0]
            data_Y = data_list_Y[0]

        _ridx = np.random.permutation(data_X.shape[0])

        X = data_X[_ridx,:,:,:,:]
        Y = data_Y[_ridx,:,:]


        if self.residual_mode:
            if self.verbose > 1:
                print('residual mode. train on (zero, low - zero, full - zero)')
            X[:,:,1,:,:] -= X[:,:,1,:,:]
            Y -= X[:,h,0,:,:]

        X = np.transpose(np.reshape(X, (X.shape[0], -1, X.shape[3], X.shape[4])), (0, 2, 3, 1))
        Y = np.reshape(Y, (Y.shape[0], Y.shape[1], Y.shape[2], 1))

        if self.verbose > 1:
            print('X, Y sizes = ', X.shape, Y.shape)


        return X, Y
