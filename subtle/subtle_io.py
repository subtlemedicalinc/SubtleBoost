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
import numpy as np
import os
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

def load_npy_file(npy_file):

    ''' Load single npy file and transpose

    Parameters:
    -----------
    npy_file : string
        name of npy file

    Returns:
    --------
    out : numpy array
        numpy array transposed for network format
    '''

    return np.transpose(np.load(npy_file), (0, 2, 3, 1))

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

def get_num_slices(npy_file, axis=0):
    f = np.load(npy_file, mmap_mode='r')
    return f.shape[axis]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, npy_list, batch_size=8, num_channel_input=1, num_channel_output=1, img_rows=128, img_cols=128, shuffle=True, verbose=True, residual_mode=True):

        'Initialization'
        self.npy_list = npy_list
        self.batch_size = batch_size
        self.num_channel_input = num_channel_input
        self.num_channel_output = num_channel_output
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.shuffle = shuffle
        self.verbose = verbose
        self.residual_mode = residual_mode
        self.current_slice = 0

        self.num_slices_per_file = np.array([get_num_slices(npy_file) for npy_file in npy_list])
        self.cumsum_slices = np.cumsum(self.num_slices_per_file)
        self.num_slices = np.sum(self.num_slices_per_file)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_slices / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # figure out which file to start reading from
        file_1_idx = np.argmax(self.current_slice < self.cumsum_slices)
        print('File:', self.npy_list[file_1_idx], 'num slices:', self.num_slices_per_file[file_1_idx]) 

        start_idx = self.num_slices_per_file[file_1_idx] - self.cumsum_slices[file_1_idx] + self.current_slice

        print('current slice:', self.current_slice, 'start_idx:', start_idx)

        # figure out if we need to read slices from the next adjacent file
        remaining_slices = self.num_slices_per_file[file_1_idx] - start_idx
        print('total slices in file: ', self.num_slices_per_file[file_1_idx])
        print('remaining slices:', remaining_slices)
        if remaining_slices < self.batch_size:
            file_2_idx = file_1_idx + 1
            npy_dict = {
                    self.npy_list[file_1_idx]: np.arange(start_idx, remaining_slices),
                    self.npy_list[file_2_idx]: np.arange(0, self.batch_size - remaining_slices)
                    }
        else:
            npy_dict = {self.npy_list[file_1_idx]: np.arange(start_idx, start_idx + self.batch_size)}

        print('list of slices:', npy_dict)

        # Generate data
        X, Y = self.__data_generation(npy_dict)

        self.current_slice = self.current_slice + self.batch_size

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.npy_list))
        if self.shuffle == True:
            _ridx = np.random.permutation(len(self.npy_list))
            self.indexes = self.indexes[_ridx]
            self.num_slices_per_file = self.num_slices_per_file[_ridx]
            self.cumsum_slices = np.cumsum(self.num_slices_per_file)
        self.current_slice = 0

    def __data_generation(self, npy_dict):
        'Generates data containing batch_size samples' 

        # load volumes
        # each element of the data_list contains 3 sets of 3D
        # volumes containing zero, low, and full contrast.
        # the number of slices may differ but the image dimensions
        # should be the same

        print(npy_dict)

        data_list = []
        for k in npy_dict.keys():
            print(k)
            d = np.load(k, mmap_mode='r')
            data_list.append(d[npy_dict[k],:,:,:].transpose((0, 2, 3, 1)))

        if len(data_list) > 1:
            data = np.concatenate(data_list, axis=0)
        else:
            data = data_list[0]

        _ridx = np.random.permutation(data.shape[0])

        X = data[_ridx,:,:,:2]
        Y = data[_ridx,:,:,-1][:,:,:,None]

        if self.verbose:
            print('X, Y sizes = ', X.shape, Y.shape)

        if self.residual_mode:
            if self.verbose:
                print('residual mode. train on (zero, low - zero, full - zero)')
            X[:,:,:,1] -= X[:,:,:,0]
            #X[:,:,:,1] = np.maximum(0., X[:,:,:,1])
            Y -= X[:,:,:,0][:,:,:,None]
            #Y = np.maximum(0., Y)

        return X, Y
