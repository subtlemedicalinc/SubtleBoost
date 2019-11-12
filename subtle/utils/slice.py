import numpy as np
import h5py

from . import io as utils_io

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
        file_type = utils_io.get_file_type(input_file)

    if file_type == 'npy':
        return load_slices_npy(input_file, slices, dim=dim)
    elif file_type == 'h5':
        return load_slices_h5(input_file, slices, h5_key=params['h5_key'], dim=dim)
    else:
        #FIXME add real exception handling
        print('subtle_io/load_slices: ERROR. unrecognized file type', file_type)
        sys.exit(-1)

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
