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

