import numpy as np
import h5py

from . import io as utils_io

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

    data_shape = utils_io.get_shape(data_file, file_type, params)
    return data_shape[axis]


def build_slice_list(data_list, params={'h5_key': 'data'}, slice_axis=[0]):
    ''' Builds two lists where the index is the
    slice number and the value is the file name / slice index.
    The length of the list is the total number of slices

    Parameters:
    -----------
    data_list : list
        list of file names
    slice_axis : int
        axis to check for slices

    Returns:
    --------
    slice_list_files : list
        slice list corresponding to files
    slice_list_indexes : list
        slice list corresponding to slice indexes
    '''

    slice_list_files = []
    slice_list_indexes = []

    for ax in slice_axis:
        for data_file in data_list:
            num_slices = get_num_slices(data_file, params=params, axis=ax)
            slice_list_files.extend([data_file] * num_slices)
            indices = [{'index': idx, 'axis': ax} for idx in range(num_slices)]
            slice_list_indexes.extend(indices)

    return slice_list_files, slice_list_indexes
