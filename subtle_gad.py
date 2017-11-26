#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Predict enhanced gad contrast from lowcon, precon
Created on Thu Nov 16 16:50:27 2017

@author: enhao, Nov 21 2017
@Subtle Medical

Update Nov 24, 2017
1. add logger module
2. update order to 


Arguments:
	dir_precon = pre-contrast directory
	dir_lowcon = low-contrast directory
	dir_highcon = high-contrast directory, optional
	dir_output = output dicoms to new directory, 
	dir_plot = output figures, optional
	usage_gpu = '0', specify gpu usage, '' to use CPU
	process_highcon = True, can use False to skip high contrast process, can be ~1min faster
	skip_coreg = False, can use True to skip coreg, can be ~1min faster
	delete_nifti = False, can use True to delete nifti to save space, rather than keep to make future runs faster
	gpu_memory = 0.8, how much gpu memory to use
	batch_size = 4, how large batch size to use, 	

TODO:
[] use generator for output
[] add argument controls
[] add formatted logs and return error codes
[] delete nifti files

Example:


"""
'''
dependencies
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import datetime

'''
dicom
'''
# import SimpleITK as sitk
import dicom
import nibabel as nib
import dicom2nifti

'''
cafndl
'''
from cafndl.cafndl_fileio import *
from cafndl.cafndl_utils import *
from cafndl.cafndl_network import *

'''
subtle
'''
from subtle_utils import *
from subtle_log import *

'''
constant
'''
ext_nifti = '.nii.gz'
ext_dicom = '.dcm'
ext_coreg = '_coreg'

def enhance_gad(dir_precon = '../data/data_lowcon/lowcon_0006/007/',
			dir_lowcon = '../data/data_lowcon/lowcon_0006/011/',
			dir_highcon = None,
			dir_output = None, #'../data/data_lowcon/lowcon_0006/011_enhanced/',
			dir_plot = None,
			usage_gpu = '0',
			process_highcon = False,
			skip_coreg = False,
			delete_nifti = False,
			gpu_memory = 0.8,
			batch_size = 4):	

	'''
	logger
	'''
	my_logger = Logger()

	''' 
	setup gpu
	'''
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = usage_gpu
	my_logger.update_event("setting cuda devices")

	'''
	preprocessing
	'''      
	# skip high contrast if not provide
	if dir_highcon is None:
		dir_highcon = dir_lowcon
		process_highcon = False  
	# convert	
	t_convert = datetime.datetime.now()
	filepath_nifti_pre = os.path.relpath(dir_precon)+ext_nifti
	filepath_nifti_low = os.path.relpath(dir_lowcon)+ext_nifti
	filepath_nifti_high = os.path.relpath(dir_highcon)+ext_nifti
	if not os.path.exists(filepath_nifti_pre):
		dicom2nifti.convert_dicom.dicom_series_to_nifti(dir_precon, filepath_nifti_pre)
	if not os.path.exists(filepath_nifti_low):
		dicom2nifti.convert_dicom.dicom_series_to_nifti(dir_lowcon, filepath_nifti_low)
	if process_highcon:
		if not os.path.exists(filepath_nifti_high):
			dicom2nifti.convert_dicom.dicom_series_to_nifti(dir_lowcon, filepath_nifti_high)
	else:
		filepath_nifti_high = filepath_nifti_low
	t_convert_finish = datetime.datetime.now()
	print('converted dicom to nifti, take time:', t_convert_finish - t_convert)
	my_logger.log_event("converting dicom to nifti")

	if skip_coreg:
		# skip coreg
		filepath_nifti_low_coreg = filepath_nifti_low
		filepath_nifti_high_coreg = filepath_nifti_high
	else:
		# co-registration
		t_coreg = datetime.datetime.now()
		filepath_nifti_low_coreg = os.path.relpath(dir_lowcon) + ext_coreg + ext_nifti
		# conduct co-registration
		conduct_coreg(filepath_nifti_low, filepath_nifti_pre, filepath_nifti_low_coreg, overwrite=0)
		t_coreg_finish = datetime.datetime.now()
		print('co-registration {0} with {1} to {2}'.format(
			filepath_nifti_low, filepath_nifti_pre, filepath_nifti_low_coreg), 
			t_coreg_finish - t_coreg)

		# co-registration
		if process_highcon:
			t_coreg = datetime.datetime.now()
			filepath_nifti_high_coreg = os.path.relpath(dir_highcon) + ext_coreg + ext_nifti
			# conduct co-registration
			conduct_coreg(filepath_nifti_high, filepath_nifti_pre, filepath_nifti_high_coreg, overwrite=0)
			t_coreg_finish = datetime.datetime.now()
			print('co-registration {0} with {1} to {2}'.format(
				filepath_nifti_high, filepath_nifti_pre, filepath_nifti_high_coreg),
				  t_coreg_finish - t_coreg)   
		else:
			filepath_nifti_high_coreg = filepath_nifti_high

	my_logger.log_event("finishing cor-registration")


	'''
	load datasets
	'''
	# load data
	t_load_start =  datetime.datetime.now()
	data_pre = nib.load(filepath_nifti_pre).get_data()
	data_low = nib.load(filepath_nifti_low_coreg).get_data()
	if process_highcon:
		data_high = nib.load(filepath_nifti_high_coreg).get_data()
	else:
		data_high = data_low
	# print(data_pre.shape, data_low.shape, data_high.shape)
	t_load_finish =  datetime.datetime.now()
	print('load data', t_load_finish - t_load_start)
	# check dimensions
	nx, ny, nz = data_pre.shape
	assert nz == data_low.shape[-1]
	assert nz == data_high.shape[-1]
	print('dimension:', nx, ny, nz)
	my_logger.log_event("finishing load data")

	# delete nifti
	if delete_nifti:
		my_logger.log_event("deleting nifti: TBD")


	'''
	adjust scaling
	'''

	# scaling
	scale_pre = np.mean(data_pre.flatten())
	scale_low = np.mean(data_low.flatten())
	data_low = data_low / scale_low
	data_pre = data_pre / scale_pre
	if process_highcon:
		scale_high = np.mean(data_high.flatten())	
		data_high = data_high / scale_high
	else:
		data_high = data_low
	t_scale_start = datetime.datetime.now()

	# compute best scale
	list_scale = np.linspace(0.5,1.5,30)
	cost_func = lambda x: np.mean(np.abs(x[np.abs(x)>0.1].flatten()))
	best_scale_low = adjust_scale_for_cost(data_low[:,:,(nz/2-5):(nz/2+5)], data_pre[:,:,(nz/2-5):(nz/2+5)],
											list_scale = list_scale,
											iteration = 2)
	if process_highcon:
		best_scale_high = adjust_scale_for_cost(data_high[:,:,(nz/2-5):(nz/2+5)], data_pre[:,:,(nz/2-5):(nz/2+5)],
											list_scale = list_scale,
											iteration = 2)		
	else:
		best_scale_high	= best_scale_low
	print('best scaling for low and post:', best_scale_low, best_scale_high)
	
	# scaling
	data_low_adjust = data_low * best_scale_low
	data_high_adjust = data_high * best_scale_high
	contrast_low_adjust = data_low_adjust - data_pre
	contrast_high_adjust = data_high_adjust - data_pre	
	
	# timing
	t_scale_finish = datetime.datetime.now()
	print('scales adjusted', t_scale_finish - t_scale_start)
	my_logger.log_event("finishing scaling differet series")

	'''
	clean up unused variables
	'''
	del data_low
	del data_high
	del data_low_adjust
	del data_high_adjust
	my_logger.log_event("finishing scaling differet series, unused var deleted")

	''' 
	setup models
	'''
	# scaling
	scale_contrast_high_vs_low = 0.17
	scale_data = 0.5
	scale_baseline = 1.0
	# checkpoints
	filename_checkpoint = '../ckpt/model_with_generator_1120.ckpt'
	# related to model
	num_poolings = 3
	num_conv_per_pooling = 3
	# related to training
	validation_split = 0.1
	batch_size = batch_size#4
	# default settings
	keras_memory = gpu_memory#0.9
	keras_backend = 'tf'
	with_batch_norm = True
	y_range = [-1,1]
	# dimension
	num_channel_input = 5*2
	num_channel_output = 1
	img_rows = 512
	img_cols = 512
	print('setup parameters')

	# startup model
	clearKerasMemory()
	setKerasMemory(keras_memory)
	model = deepEncoderDecoder(num_channel_input = num_channel_input,
							num_channel_output = num_channel_output,
							img_rows = img_rows,
							img_cols = img_cols,
							num_poolings = num_poolings, 
							num_conv_per_pooling = num_conv_per_pooling, 
							with_bn = with_batch_norm, verbose=1,
							y=np.array(y_range))
	print('train model:', filename_checkpoint)
	print('parameter count:', model.count_params())

	#load model
	t_load_model_start = datetime.datetime.now()
	model.load_weights(filename_checkpoint)
	t_load_model_finish = datetime.datetime.now()
	print('load model from {0} taking {1}'.format(filename_checkpoint, t_load_model_finish-t_load_model_start))
	my_logger.log_event("finishing model initialization")

	'''
	prepare inputs to model
	'''
	t_concate_start = datetime.datetime.now()
	nz_25d = 5
	nz_25d_half = nz_25d/2
	list_im = []
	for i in xrange(nz_25d_half,nz-nz_25d_half):
		# get low contrast enhance
		im_lowcon = contrast_low_adjust[:,:,(i-nz_25d_half):(i+nz_25d_half+1)]
		# get pre contrast
		im_pre = data_pre[:,:,(i-nz_25d_half):(i+nz_25d_half+1)]    
		# post contrast enhance, scaled
		# im_highcon = contrast_high_adjust[:,:,i:(i+1)]*scale_contrast_high_vs_low   

		# concat
		im_input = np.concatenate([im_lowcon, im_pre], axis=-1)
		list_im.append(im_input[np.newaxis,:,:,:])
	# concatenation
	data_input = np.concatenate(list_im, axis=0) 
	data_input *= scale_data
	contrast_low_avg = np.mean(data_input[:,:,:,:nz_25d], axis=-1)[:,:,:,np.newaxis]
	t_concate_finish = datetime.datetime.now()    
	print('concate {0} images using time {1}'.format(nz-nz_25d, t_concate_finish-t_concate_start))	
	my_logger.log_event("finishing preparing data for predicting")

	'''
	predict
	'''
	t_predict_start = datetime.datetime.now()
	try:
		contrast_low_residual = model.predict(data_input, batch_size = batch_size, verbose = 1)
		mix_avg = 0.1
	except:
		print('error in prediction, will use avg results')
		contrast_low_residual = data_input[:,:,:,2:3]
		mix_avg = 1.0
	t_predict_finish = datetime.datetime.now()
	print('predict for data size {0}, time {1}'.format(data_input.shape, t_predict_finish - t_predict_start))
	my_logger.log_event("finishing predicting data")


	'''
	post-proceesing
	'''
	contrast_low_avg2 = contrast_low_adjust * 1.0
	contrast_low_avg2[:,:,nz_25d_half:(nz-nz_25d_half)] = np.squeeze(contrast_low_avg.transpose([1,2,0,3])) * 1.0
	contrast_low_enhanced_from_avg = contrast_low_avg2 * scale_baseline
	# if use new model from avg contrast: 
	contrast_low_enhanced = contrast_low_adjust * scale_baseline
	contrast_low_enhanced[:,:,nz_25d_half:(nz-nz_25d_half)] += np.squeeze(contrast_low_residual.transpose([1,2,0,3]))
	contrast_low_enhanced = contrast_low_enhanced * (1-mix_avg) + contrast_low_enhanced_from_avg * mix_avg
	my_logger.log_event("finishing mixing contrast predictions")

	'''
	visualization
	'''
	if dir_plot is not None:
		if not os.path.exists(dir_plot):
			os.mkdir(dir_plot)
		for index_slice in xrange(10,nz,10):
			list_concat = [
							contrast_low_adjust[:,:,index_slice]*(contrast_low_adjust[:,:,index_slice]>0),
							contrast_low_avg2[:,:,index_slice]*(contrast_low_avg2[:,:,index_slice]>0),
							contrast_low_enhanced[:,:,index_slice]*(contrast_low_enhanced[:,:,index_slice]>0),
							contrast_high_adjust[:,:,index_slice]*(contrast_high_adjust[:,:,index_slice]>0)*scale_contrast_high_vs_low,
							]
			list_concat = [np.flipud(x.T) for x in list_concat]
			# list_concat = [np.abs(x[300:450,200:350]) for x in list_concat]
			im_concat = np.concatenate(list_concat, axis = 1)
			plt.figure(figsize = [20,8])
			plt.imshow(im_concat, 
					   clim = [0,2],
					   cmap = 'gray')
			plt.axis('off')
			filename_plot = 'contrast_slice_{0}.jpg'.format(index_slice)
			print('export to {0}'.format(filename_plot))
			plt.savefig(os.path.join(dir_plot,filename_plot))
		my_logger.log_event("finishing plot figures")


	'''
	scale contrast
	'''
	threshold_percentile = 0.1
	threshold_percentile_value_low = 0.5
	threshold_percentile_value_high = 4.0
	threshold_percentile_value_output = [6.0, 7.0]
	threshold_clamp_contrast = 0.01
	percentile_use = 95
	percentile_high_adjust = np.percentile(contrast_high_adjust[:,:,(nz/4):(nz*3/4)][contrast_high_adjust[:,:,(nz/4):(nz*3/4)]>threshold_percentile].flatten(), 
										   percentile_use)
	percentile_low_adjust = np.percentile(contrast_low_adjust[:,:,(nz/4):(nz*3/4)][contrast_high_adjust[:,:,(nz/4):(nz*3/4)]>threshold_percentile].flatten(), 
										  percentile_use)
	percentile_low_enhanced = np.percentile(contrast_low_enhanced[:,:,(nz/4):(nz*3/4)][contrast_low_enhanced[:,:,(nz/4):(nz*3/4)]>threshold_percentile].flatten(), 
										   percentile_use)
	print('percentile:', percentile_low_adjust, percentile_high_adjust, percentile_low_enhanced)
	# print  # goal is to change the percentail to 5
	# print np.percentile(contrast_low_adjust[contrast_high_adjust>0.001].flatten(), 95)
	# low dose should be around 0.5-2, high dose should be around 4-10, target 5
	# assert(percentile_low_adjust>threshold_percentile_value_low)
	# assert(percentile_high_adjust>threshold_percentile_value_high)
	contrast_low_adjust[contrast_low_adjust < threshold_clamp_contrast] = 0.0
	percentile_high_adjust = max(threshold_percentile_value_output[1], 
								min(threshold_percentile_value_output[0], percentile_high_adjust))
	data_low_enhanced = contrast_low_adjust * percentile_high_adjust / percentile_low_enhanced + data_pre
	data_low_enhanced = np.maximum(0.0, data_low_enhanced)
	my_logger.log_event("finishing rescaling predicted contrasts")

	'''
	export DICOM
	'''
	if dir_output is not None:		
	
		seriesdescription = 'Contrast Enhanced MRI DL'
		list_filename_dicom = sorted([x for x in os.listdir(dir_precon) if x.endswith(ext_dicom)])
		time_export_dicom = datetime.datetime.now()
		if not os.path.exists(dir_output):
			os.mkdir(dir_output)
		for i in xrange(nz):
			# get original dicom
			filename_dicom = list_filename_dicom[i]
			filepath_dicom = os.path.join(dir_precon, filename_dicom)
			file_dicom = dicom.read_file(filepath_dicom)
			
			# replace pixel
			pixel_value = (data_low_enhanced[:,:,i] * scale_pre).astype(np.int16)
			file_dicom.PixelData=pixel_value.tostring()
			file_dicom.SeriesDescription=seriesdescription
			
			# save
			filepath_output = os.path.join(dir_output, filename_dicom)
			dicom.write_file(filepath_output, file_dicom)
			
			if i%100 == 0:
				print('export to {0}'.format(filepath_output))

		time_export_dicom_finish = datetime.datetime.now()        
		print('export {0} files using {1}'.format(nz, time_export_dicom_finish-time_export_dicom))
		my_logger.log_event("finishing exporting dicom files")

	return (data_low_enhanced * scale_pre).astype(np.int16)

if __name__ == '__main__':
	'''
	Main default input
	'''
	dir_precon = '../data/data_lowcon/lowcon_0006/007/'
	dir_lowcon = '../data/data_lowcon/lowcon_0006/011/'
	dir_highcon = '../data/data_lowcon/lowcon_0006/014/'
	dir_output = '../data/data_lowcon/lowcon_0006/011_enhanced/'
	dir_plot = '../data/data_lowcon/lowcon_0006/results'
	enhance_gad(dir_precon, dir_lowcon, dir_highcon)


