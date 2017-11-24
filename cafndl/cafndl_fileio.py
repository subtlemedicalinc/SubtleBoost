import dicom
import nibabel as nib
import numpy as np
import scipy.io as sio
import os
from cafndl_utils import augment_data

# support data augmentations,
# scale with fixed value or normalized using norm
# support select subsets of slices

def prepare_data_from_nifti(path_load, list_augments=[], scale_by_mean=True, slices=None, scale_factor=-1):
	# get nifti
	nib_load = nib.load(path_load)
	print(nib_load.header)
	
	# get data
	data_load = nib_load.get_data()
	if np.ndim(data_load)==3:
		data_load = data_load[:,:,:,np.newaxis]	

	# transpose to slice*x*y*channel
	data_load = np.transpose(data_load, [2,0,1,3])
	
	# scale
	if scale_by_mean:
		data_load /= np.mean(data_load.flatten()) + 0.0
	if scale_factor>0:
		data_load *= scale_factor + 0.0
	print('scale to with maximum value of {0}'.format(np.max(data_load.flatten())))

	# extract slices
	if slices is not None:
		data_load = data_load[np.array(slices),:,:,:]	
		if np.ndim(data_load)==3:
			data_load = data_load[np.newaxis,:,:,:]	
	
	# finish loading data
	print('loaded from {0}, data size {1} (sample, x, y, channel)'.format(path_load, data_load.shape))    
	
	# augmentation
	if len(list_augments)>0:
		print('data augmentation')
		list_data = []
		for augment in list_augments:
			print(augment)
			data_augmented = augment_data(data_load, axis_xy = [1,2], augment = augment)
			list_data.append(data_augmented.reshape(data_load.shape))
		data_load = np.concatenate(list_data, axis = 0)
	return data_load

def prepare_data_from_mat(path_load, list_augments=[], scale_by_mean=True, slices=None, scale_factor=-1, augment_25D=0, channel_as_contrast=False):
	# get nifti
	data_load = sio.loadmat(path_load)
	key_var = [x for x in data_load.keys() if not x.startswith('_')][0]
	data_load = data_load[key_var]
	
	# reshape
	if np.ndim(data_load)==3:
		data_load = data_load[:,:,:,np.newaxis]	

	# transpose to slice*x*y*channel
	data_load = np.transpose(data_load, [2,0,1,3])
	
	# scale
	if scale_by_mean:
		data_load /= np.mean(data_load.flatten()) + 0.0
	if scale_factor>0:
		data_load *= scale_factor + 0.0
	print('scale to with maximum value of {0}'.format(np.max(data_load.flatten())))

	# augment 2.5D
	if augment_25D==0:
		data_load = data_load[:,:,:,:,np.newaxis]
	else:
		data_load_augment = np.array(data_load)[:,:,:,:,np.newaxis]
		for i_augment in xrange(1,augment_25D+1):
			data_load_shift = np.array(data_load)
			data_load_shift[i_augment:,:,:,:]=data_load_shift[:-i_augment,:,:,:]
			data_load_shift = data_load_shift[:,:,:,:,np.newaxis]
			data_load_augment = np.concatenate([data_load_augment, data_load_shift], axis=-1)
		for i_augment in xrange(1,augment_25D+1):
			data_load_shift = np.array(data_load)
			data_load_shift[:-i_augment:,:,:,:]=data_load_shift[i_augment:,:,:,:]
			data_load_shift = data_load_shift[:,:,:,:,np.newaxis]
			data_load_augment = np.concatenate([data_load_augment, data_load_shift], axis=-1)
		data_load = data_load_augment

	# extract slices
	if slices is not None:
		data_subslice = data_load[np.array(slices),:,:,:,:]	
		if np.ndim(data_subslice)<np.ndim(data_load):
			data_subslice = data_subslice[np.newaxis,:,:,:,:]				
		data_load = data_subslice
	
	# finish loading data
	print('loaded from {0}, data size {1} (sample, x, y, channel, 25D)'.format(path_load, data_load.shape))    
	
	# augmentation
	if len(list_augments)>0:
		print('data augmentation')
		list_data = []
		for augment in list_augments:
			print(augment)
			data_augmented = augment_data(data_load, axis_xy = [1,2], augment = augment)
			list_data.append(data_augmented.reshape(data_load.shape))
		data_load = np.concatenate(list_data, axis = 0)


	# transpose
	if channel_as_contrast:
		data_load = np.reshape(data_load, [data_load.shape[0], data_load.shape[1], data_load.shape[2], data_load.shape[3]*data_load.shape[4]])
	else:
		data_load = np.transpose(data_load, [0,3,1,2,4])
		data_load = np.reshape(data_load, [data_load.shape[0]*data_load.shape[1], data_load.shape[2], data_load.shape[3], data_load.shape[4]])
	return data_load


def prepare_data_from_dicom_folder(path_load, ext_dicom = 'dcm', key_sort = lambda x:x,
									dim_reshape=None,
									list_augments=[], slices=None,
									scale_by_mean=True, scale_method=lambda x:np.mean(x), scale_factor=-1, 
									augment_25D=0, channel_as_contrast=True):

	# get list of file
	list_file = sorted([x for x in os.listdir(path_load) if x.find(ext_dicom)>=0], 
						key = key_sort)#lambda x:int(x.split('.')[-1]))
	print(list_file[:10])
	list_filepath = [os.path.join(path_load, x) for x in list_file]
	print('load {0} dicom files'.format(len(list_filepath)))

	# load into data
	list_data_load = []
	for filepath in list_filepath:
		file_dicom = dicom.read_file(filepath)
#         print(file_dicom.keys())
		data_slice = file_dicom.pixel_array
		list_data_load.append(data_slice[:,:,np.newaxis])
	data_load = np.concatenate(list_data_load, axis=-1)
	print('raw dimension:', data_load.shape)
	
	# reshape
	if dim_reshape is not None:
		if type(dim_reshape)==list:
			data_load = data_load.reshape(dim_reshape)
		else:
			dim_reshape = np.shape(data_load)
			dim_reshape.append(dim_reshape)
			dim_reshape[-2]/=dim_reshape
			data_load = data_load.reshape(dim_reshape)
	if np.ndim(data_load)==3:
		data_load = data_load[:,:,:,np.newaxis]	

	# transpose to slice*x*y*channel
	data_load = np.transpose(data_load, [2,0,1,3]).astype(np.float32)

	# scale
	if scale_by_mean:
		value_scale = scale_method(data_load.flatten()) + 0.0 
		data_load /= value_scale + 0.0
		print('scale to with mean value of {0}'.format(value_scale))
	if scale_factor>0:
		data_load *= scale_factor + 0.0
		print('scale by value {0}'.format(scale_factor))
		

	# augment 2.5D
	if augment_25D==0:
		data_load = data_load[:,:,:,:,np.newaxis]
	else:
		data_load_augment = np.array(data_load)[:,:,:,:,np.newaxis]
		for i_augment in xrange(1,augment_25D+1):
			data_load_shift = np.array(data_load)
			data_load_shift[i_augment:,:,:,:]=data_load_shift[:-i_augment,:,:,:]
			data_load_shift = data_load_shift[:,:,:,:,np.newaxis]
			data_load_augment = np.concatenate([data_load_augment, data_load_shift], axis=-1)
		for i_augment in xrange(1,augment_25D+1):
			data_load_shift = np.array(data_load)
			data_load_shift[:-i_augment:,:,:,:]=data_load_shift[i_augment:,:,:,:]
			data_load_shift = data_load_shift[:,:,:,:,np.newaxis]
			data_load_augment = np.concatenate([data_load_augment, data_load_shift], axis=-1)
		data_load = data_load_augment

	# extract slices
	if slices is not None:
		data_subslice = data_load[np.array(slices),:,:,:,:]	
		if np.ndim(data_subslice)<np.ndim(data_load):
			data_subslice = data_subslice[np.newaxis,:,:,:,:]				
		data_load = data_subslice

	# finish loading data
	print('loaded from {0}, data size {1} (sample, x, y, channel, 25D)'.format(path_load, data_load.shape))    

	# augmentation
	if len(list_augments)>0:
		print('data augmentation')
		list_data = []
		for augment in list_augments:
			print(augment)
			data_augmented = augment_data(data_load, axis_xy = [1,2], augment = augment)
			list_data.append(data_augmented.reshape(data_load.shape))
		data_load = np.concatenate(list_data, axis = 0)


	# transpose
	if channel_as_contrast:
		data_load = np.reshape(data_load, [data_load.shape[0], data_load.shape[1], data_load.shape[2], data_load.shape[3]*data_load.shape[4]])
	else:
		data_load = np.transpose(data_load, [0,3,1,2,4])
		data_load = np.reshape(data_load, [data_load.shape[0]*data_load.shape[1], data_load.shape[2], data_load.shape[3], data_load.shape[4]])
	return data_load		

