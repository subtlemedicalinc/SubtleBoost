"""
train Deep Magic model
1014
Updated: 
1) use the 6 map
1015
Updated:
1) no slice augment for now
2) correct the dimensions
3) ignore STL contrast
4) add normalization
1016
1) re-organize code
2) scaling 0.1
3) add 6 contrasts
4) add different scale for gt
1021
1) use generator
2) use 3 datasets
"""
''' dependencies '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import json
import datetime

from cafndl.cafndl_fileio import *
from cafndl.cafndl_utils import *
from cafndl.cafndl_network import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

''' data info '''
filename_checkpoint = '../ckpt/model_1021_generator.ckpt'
filename_init = ''
list_dataset_train =  [
				{
				 'inputs':[
						   '../data/MAGiC_data_for_DL/magic_6139/s32741_T2FLAIR/',
						   '../data/MAGiC_data_for_DL/magic_6139/s32740_T1/',
						   '../data/MAGiC_data_for_DL/magic_6139/s32742_T2/',
						   '../data/MAGiC_data_for_DL/magic_6139/s32733_PD/',                     
						   '../data/MAGiC_data_for_DL/magic_6139/s32743_T1FLAIR/', 
						   '../data/MAGiC_data_for_DL/magic_6139/s32736/',                      
						   ],
				 'gt':'../data/MAGiC_data_for_DL/ex6139_origflair/s32739/',
				 'nslice':28,
				 # 'slices':np.array(range(0,28,2)).tolist()
				 '25D':0,
				},
				{
				 'inputs':[
						   '../data/MAGiC_data_for_DL/magic_8582/s40041_T2FLAIR/',
						   '../data/MAGiC_data_for_DL/magic_8582/s8088_T1/',                     
						   '../data/MAGiC_data_for_DL/magic_8582/s8086_T2/',
						   '../data/MAGiC_data_for_DL/magic_8582/s8081_PD/',                     
						   '../data/MAGiC_data_for_DL/magic_8582/s8087_T1FLAIR/',                                         
						   '../data/MAGiC_data_for_DL/magic_8582/s8082/',                     
						   ],
				 'gt':'../data/MAGiC_data_for_DL/ex8582_origflair/s8085/',
				 'nslice':28,
				 # 'slices':np.array(range(0,28,2)).tolist()
				 '25D':0,                    
				},
				{
				 'inputs':[
						   '../data/MAGiC_data_for_DL/magic_8806/s5238_T2FLAIR/',
						   '../data/MAGiC_data_for_DL/magic_8806/s5244_T1/',                     
						   '../data/MAGiC_data_for_DL/magic_8806/s5240_T2/',                     
						   '../data/MAGiC_data_for_DL/magic_8806/s5242_PD/',                                          
						   '../data/MAGiC_data_for_DL/magic_8806/s5250_T1FLAIR/', 
						   '../data/MAGiC_data_for_DL/magic_8806/s5245/',                     
						   ],
				 'gt':'../data/MAGiC_data_for_DL/ex8806_origflair/s5241/',
				 'nslice':30,
				 '25D':0,    
				 },
				] 		
num_dataset_train = len(list_dataset_train)                
print('process {0} data description'.format(num_dataset_train))

'''
augmentation
'''
list_augments = []
num_augment_flipxy = 2
num_augment_flipx = 2
num_augment_flipy = 2
num_augment_shiftx = 1
num_augment_shifty = 1
for flipxy in range(num_augment_flipxy):
	for flipx in range(num_augment_flipx):
		for flipy in range(num_augment_flipy):
			for shiftx in range(num_augment_shiftx):
				for shifty in range(num_augment_shifty):
					augment={'flipxy':flipxy,'flipx':flipx,'flipy':flipy,'shiftx':shiftx,'shifty':shifty}
					list_augments.append(augment)
num_augment=len(list_augments)
print('will augment data with {0} augmentations'.format(num_augment))

'''
file loading related 
'''
ext_dicom = 'MRDC'
key_sort = lambda x: int(x.split('.')[-1])
scale_method = lambda x:np.mean(np.abs(x))
scale_by_mean = False
scale_factor = 1/32768.
ext_data = 'npz'
dir_samples = '../data/data_sample/'

def export_data_to_npz(data_train_input, data_train_gt,dir_numpy_compressed, index_sample_total=0, ext_data = 'npz'):  
	index_sample_accumuated = index_sample_total
	num_sample_in_data = data_train_input.shape[0]
	if not os.path.exists(dir_numpy_compressed):
		os.mkdir(dir_numpy_compressed)
		print('create directory {0}'.format(dir_numpy_compressed))
	print('start to export data dimension {0}->{1} to {2} for index {3}', 
		  data_train_input.shape, data_train_gt.shape, dir_numpy_compressed, 
		  index_sample_total)        
	for i in xrange(num_sample_in_data):
		im_input = data_train_input[i,:]
		im_output = data_train_gt[i,:]
		filepath_npz = os.path.join(dir_numpy_compressed,'{0}.{1}'.format(index_sample_accumuated, ext_data))
		with open(filepath_npz,'w') as file_input:
			np.savez_compressed(file_input, input=im_input, output=im_output)
		index_sample_accumuated+=1
	print('exported data dimension {0}->{1} to {2} for index {3}', 
		  data_train_input.shape, data_train_gt.shape, dir_numpy_compressed, 
		  [index_sample_total,index_sample_accumuated])
	return index_sample_accumuated

'''
generate train data
'''
# scale differently for gt such that synthesized FLAIR and real FLAIR match in scale
scale_by_mean_gt = True
scale_factor_gt = 0.1
list_train_input = []
list_train_gt = []        
index_sample_total = 0
for index_data in range(num_dataset_train):
	# directory
	list_data_train_input = []
	try:
		slices = np.array(list_dataset_train[index_data]['slices'])
		nslice = len(slices)
	except:
		slices = None
		nslice = 0
		
	# dim for training
	try:
		nslice = list_dataset_train[index_data]['nslice']
		dim_reshape = [512, 512, nslice, 1]
	except:
		nslice = 0
		dim_reshape = None

	# 2.5D
	try:
		augment_25D = list_dataset_train[index_data]['25D']
	except:
		augment_25D = 0
			  
	

	# get inputs
	for path_train_input in list_dataset_train[index_data]['inputs']:
		# load data		
			# scale differently for ground-truth and baseline
		if len(list_data_train_input)==0:
			data_train_input = prepare_data_from_dicom_folder(path_load = path_train_input, list_augments=list_augments, 
														  ext_dicom = ext_dicom, key_sort = key_sort,
														  dim_reshape = dim_reshape,
														  scale_by_mean = scale_by_mean_gt, scale_method = scale_method, scale_factor = scale_factor_gt, 
														  slices = slices, augment_25D=augment_25D)        
		else:
			data_train_input = prepare_data_from_dicom_folder(path_load = path_train_input, list_augments=list_augments, 
														  ext_dicom = ext_dicom, key_sort = key_sort,
														  dim_reshape = dim_reshape,
														  scale_by_mean = scale_by_mean, scale_method = scale_method, scale_factor = scale_factor, 
														  slices = slices, augment_25D=augment_25D)
		# data_train_input = prepare_data_from_nifti(path_train_input, list_augments, scale_by_mean, slices)
		print(path_train_input, data_train_input.shape)
		list_data_train_input.append(data_train_input)
	# print('shape:', [x.shape for x in list_data_train_input])
	data_train_input = np.concatenate(list_data_train_input, axis=3)

	# get ground truth
	path_train_gt = list_dataset_train[index_data]['gt']
	data_train_gt = prepare_data_from_dicom_folder(path_load = path_train_gt, list_augments=list_augments, 
												   ext_dicom = ext_dicom, key_sort = key_sort,
												   scale_by_mean = scale_by_mean_gt, scale_method = scale_method, scale_factor = scale_factor_gt, 
												   slices=slices, augment_25D=0)

	# # append
	# list_train_input.append(data_train_input)
	# list_train_gt.append(data_train_gt)

	# export
	index_sample_total = export_data_to_npz(data_train_input, 
											data_train_gt,
											dir_samples, 
											index_sample_total, 
											ext_data)

''''
setup model
'''
# related to model
num_poolings = 1
num_conv_per_pooling = 1
# related to training
validation_split = 0.1
batch_size = 4
# default settings
keras_memory = 0.4
keras_backend = 'tf'
with_batch_norm = True
y_range = [-0.5,0.5]
# dimension
num_channel_input = data_train_input.shape[-1]
num_channel_output = data_train_gt.shape[-1]
img_rows = data_train_input.shape[1]
img_cols = data_train_gt.shape[1]
print('setup parameters')


'''
init model
'''
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



'''
define generator
'''
# details inside generator
params_generator = {'dim_x': img_rows,
		  'dim_y': img_cols,
		  'dim_z': num_channel_input,
		  'dim_output': num_channel_output,
		  'batch_size': 4,
		  'shuffle': True,
		  'verbose': 0,
		  'scale_data': 1.0,
		  'scale_baseline': 1.0}
print('generator parameters:', params_generator)

class DataGenerator(object):
	'Generates data for Keras'
	def __init__(self, dim_x = 512, dim_y = 512, dim_z = 6, dim_output = 1, 
				batch_size = 2, shuffle = True, verbose = 1,
				scale_data = 1.0, scale_baseline = 1.0):
		'Initialization'
		self.dim_x = dim_x
		self.dim_y = dim_y
		self.dim_z = dim_z
		self.dim_output = dim_output
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.verbose = verbose
		self.scale_data = scale_data
		self.scale_baseline = scale_baseline

	def generate(self, dir_sample, list_IDs):
		'Generates batches of samples'
		# Infinite loop
		while 1:
			# Generate order of exploration of dataset
			indexes = self.__get_exploration_order(list_IDs)
			if self.verbose>0:
				print('indexes:', indexes)
			# Generate batches
			imax = int(len(indexes)/self.batch_size)
			if self.verbose>0:            
				print('imax:', imax)
			for i in range(imax):
				# Find list of IDs
				list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
				if self.verbose>0:
					print('list_IDs_temp:', list_IDs_temp)
				# Generate data
				X, Y = self.__data_generation(dir_sample, list_IDs_temp)
				if self.verbose>0:                
					print('generated dataset size:', X.shape, Y.shape)

				yield X, Y

	def __get_exploration_order(self, list_IDs):
		'Generates order of exploration'
		# Find exploration order
		indexes = np.arange(len(list_IDs))
		if self.shuffle == True:
			  np.random.shuffle(indexes)

		return indexes

	def __data_generation(self, dir_sample, list_IDs_temp, ext_data = 'npz'):
		'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
		# Initialization
		X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
		Y = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_output, 1))

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store volume
			data_load = np.load(os.path.join(dir_sample, '{0}.{1}'.format(ID,ext_data)))
			X[i, :, :, :, 0] = data_load['input']
			Y[i, :, :, :, 0] = data_load['output'] 
		X = X[:,:,:,:,0]
		Y = Y[:,:,:,:,0]        
		X = X * self.scale_data
		Y = Y * self.scale_data
		Y = Y - self.scale_baseline * X[:,:,:,0:1]      
		return X, Y

''' 
setup train and val generator
'''
validation_split = 0.1
index_sample_total = len([x for x in os.listdir(dir_samples) if x.endswith(ext_data)])
list_indexes_train = np.random.permutation(index_sample_total)
if validation_split>1:
	list_indexes_val = list_indexes_train[-validation_split:].tolist()
	list_indexes_train = list_indexes_train[:int(index_sample_total-validation_split)].tolist()    
else:
	list_indexes_val = list_indexes_train[-int(index_sample_total*validation_split):].tolist()
	list_indexes_train = list_indexes_train[:int(index_sample_total*(1-validation_split))].tolist()
print('train on {0} samples and validation on {1} samples'.format(
		len(list_indexes_train), len(list_indexes_val)))
training_generator = DataGenerator(**params_generator).generate(dir_samples, list_indexes_train)
validation_generator = DataGenerator(**params_generator).generate(dir_samples, list_indexes_val)



'''
setup learning 
'''
# hyper parameter in each train iteration
list_hyper_parameters=[{'lr':0.001,'epochs':50},{'lr':0.0002,'epochs':50},{'lr':0.0001,'epochs':30}]
type_activation_output = 'linear'


'''
training
'''
index_hyper_start = 0
num_hyper_parameter = len(list_hyper_parameters)
for index_hyper in xrange(index_hyper_start, num_hyper_parameter):
	hyper_train = dict(list_hyper_parameters[index_hyper])
	print('hyper parameters:', hyper_train)
	# init
	if hyper_train.has_key('init'):
		try:
			model.load_weights(hyper_train['init'])
		except:
			hyper_train['init'] = ''
			print('failed to learn from init-point ' + hyper_train['init'])
			pass
	else:
		# load previous optimal
		try:
			model.load_weights(filename_checkpoint)
			hyper_train['init'] = filename_checkpoint				
			print('model finetune from ' + filename_checkpoint)   
		except:
			hyper_train['init'] = ''
			print('failed to learn from checkpoint ' + hyper_train['init'])     
			pass
	# update filename and checkpoint
	if hyper_train.has_key('ckpt'):
		filename_checkpoint = hyper_train['ckpt']
	else:
		hyper_train['ckpt'] = filename_checkpoint
	model_checkpoint = ModelCheckpoint(filename_checkpoint, monitor='val_loss', save_best_only=True)		

	# update leraning rate
	if hyper_train.has_key('lr'):
		model.optimizer = Adam(lr=hyper_train['lr'])
	else:
		hyper_train['lr'] = -1 #default

	
	# update epochs
	if hyper_train.has_key('epochs'):
		epochs = hyper_train['epochs']
	else:
		hyper_train['epochs'] = 50
		epochs = 50

	# update train_list
	hyper_train['list_dataset_train'] = list_dataset_train
	hyper_train['type_activation_output'] = type_activation_output
	hyper_train['y_range'] = np.array(y_range).tolist()		

	# fit data
	t_start_train = datetime.datetime.now()
	try:
		# history = model.fit(data_train_input,
		# 				data_train_residual, 
		# 				epochs=epochs, 
		# 				callbacks=[model_checkpoint],
		# 				validation_split=validation_split,
		# 				batch_size=batch_size, 
		# 				shuffle=True, 
		# 				verbose=1)

		print('train with hyper parameters:', hyper_train)
		history = model.fit_generator(
					generator = training_generator,
					steps_per_epoch = len(list_indexes_train)/batch_size,
					epochs = epochs,
					callbacks =[model_checkpoint],
					validation_data = validation_generator,
					validation_steps = len(list_indexes_val)/batch_size,
					max_q_size = 16,
					)		
	except:
		history = []
		print('break training')
		continue
	t_end_train = datetime.datetime.now()
	t_elapse = (t_end_train-t_start_train).total_seconds()
	print('finish training on {0} samples from data size {1} for {2} epochs using time {3}'.format(
			data_train_input.shape, data_train_input.shape, epochs, t_elapse))
	hyper_train['elapse'] = t_elapse
	
	'''
	save training results
	'''
	# save train loss/val loss
	fig = plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	ylim_min = min(min(history.history['loss']), min(history.history['val_loss']))
	ylim_max = max(history.history['loss'])*1.2
	plt.ylim([ylim_min, ylim_max])
	plt.legend(['train', 'test'], loc='upper left')
	path_figure = filename_checkpoint+'_{0}.png'.format(index_hyper)
	plt.savefig(path_figure)

	# save history dictionary
	dict_result = {'history':history.history, 'hyper_parameter':hyper_train}
	path_history = filename_checkpoint + '_{0}.json'.format(index_hyper)
	with open(path_history, 'w') as outfile:
		json.dump(dict_result, outfile)	




