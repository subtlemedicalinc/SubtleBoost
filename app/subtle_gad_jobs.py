"""The SubtleGAD jobs

@author: Srivathsa Pasumarthi <srivathsa@subtlemedical.com>
Copyright Subtle Medical, Inc. (https://subtlemedical.com)
Created on 2020/02/06
"""

import os
from typing import Dict, Optional, Tuple
import copy
from collections import namedtuple
from multiprocessing import Pool, Queue
import re

import numpy as np
import imp
import sigpy as sp
from scipy.ndimage.interpolation import rotate, zoom as zoom_interp
from scipy.ndimage import gaussian_filter
import GPUtil
import tempfile

from subtle.util.inference_job_utils import (
    BaseJobType, GenericInferenceModel#, DataLoader2pt5D,# set_keras_memory
)
from subtle.util.data_loader import InferenceLoader
from subtle.util.multiprocess_utils import processify
from subtle.procutil import preprocess_single_series
from subtle.dcmutil import pydicom_utils, series_utils

from subtle.procutil.registration_utils import register_im

from subtle.procutil.segmentation_utils import HDBetInMemory
from subtle.procutil.image_proc_util import clip

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
#import tensorflow.compat.v1 as tf
import tqdm
import pdb
import torch
import HD_BET
import warnings
import time
from global_variables import total_time
warnings.filterwarnings("ignore")

#torch.multiprocessing.set_start_method('spawn')

# os.environ['TF2_BEHAVIOR'] = '0'

# tf.disable_v2_behavior()
# msg = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(msg))

def _init_gpu_pool(local_gpu_q: Queue):
    """
    Local function to initialize a global variable to commonly access the GPU IDs by different
    child thread in parallel MPR processing

    :param local_gpu_q: the local GPU queue that is set with the allocated GPUs
    """
    # pylint: disable=global-variable-undefined
    global gpu_pool
    gpu_pool = local_gpu_q

# pylint: disable=too-many-instance-attributes
class SubtleGADJobType(BaseJobType):
    """The job type of SubtleGAD dose reduction"""
    default_processing_config_gad = {
        # app params
        "app_name": "SubtleGAD",
        "app_id": 4000,
        "app_version": "0.0.0",
        "min_gpu_mem_mb": 9800.0,

        # model params
        "model_id": "None",

        # general params
        "not_for_clinical_use": False,
        "metadata_comp": {},

        # preprocessing params:
        #pipeline definition - custom config 
        "model_type": "gad_process",
        "pipeline_preproc": {
            'gad_process' : {
                'STEP1' : {'op': 'MASK'},
                'STEP2' : {'op': 'SKULLSTRIP'},
                'STEP3' : {'op' : 'REGISTER'},
                'STEP4' : {'op': 'SCALEGLOBAL'}
            }
            
        },
        "pipeline_postproc": {
            'gad_process' : {
                'STEP1' : {'op' : 'CLIP'},
                'STEP2' : {'op' : 'RESCALEGLOBAL'}

            }
        },

        # manufacturer specific - these are the default values
        # "perform_noise_mask": True,
        # "noise_mask_threshold": 0.1,
        # "noise_mask_area": False,
        # "noise_mask_selem": False,
        # "perform_dicom_scaling": False,
        # "transform_type": "affine",
        # "use_mask_reg": True,
        # "histogram_matching": False,
        # "joint_normalize": False,
        # "scale_ref_zero_img": False,

        # not manufacturer specific
       # "perform_registration": True,
        #"skull_strip": True,
        #"skull_strip_union": False,
        #"skull_strip_prob_threshold": 0.5,
        #"num_scale_context_slices": 20,
        #"blur_lowdose": False,
        #"cs_blur_sigma": [0, 1.5],
        "acq_plane": "AX",
        "model_resolution": [0.5, 0.5, 0.5],

        # inference params:
        "inference_mpr": True,
        "num_rotations": 3,
        "skip_mpr": False,
        "slices_per_input": 7,
        "mpr_angle_start": 0,
        "mpr_angle_end": 90,
        "reshape_for_mpr_rotate": True,
        "num_procs_per_gpu": 2,
        "allocate_available_gpus": False,

        # post processing params
        "series_desc_prefix": "SubtleGAD:",
        "series_desc_suffix": "",
        "series_number_offset": 100
    }

    #def SubtleGADProcConfig(self):
    SubtleGADProcConfig = namedtuple(
            "SubtleGADProcConfig", ' '.join(list(default_processing_config_gad.keys()))
        )

        #return SubtleGADProcConfig

    @property
    def function_keys(self):
        """
        Dictionary defining the pipeline operations keywords and
        their associated function and default parameters.
        Format is (function, default parameters dictionary, param mapping to main config)
        :return:
        """
        function_keys = {
            'MASK' : (self._mask_noise_process,
                        {"noise_mask_area" : False,
                         "noise_mask_threshold": 0.1,
                         "noise_mask_selem": False}, {}),
            'SKULLSTRIP': (self.apply_brain_mask, {}, {}),
            'REGISTER': (self._register, 
                        {"transform_type" : "affine",
                         "use_mask_reg" : True}, {}),
            'HIST': (self._match_histogram,
                        {}, {}),
            'SCALETAG': (self._scale_intensity_with_dicom_tags,
                        {}, {}),
            'SCALEGLOBAL': (self._scale_intensity,
                        {"num_scale_context_slices" : 20,
                         "joint_normalize": False, 
                         "scale_ref_zero_img": False}, {}),
            'CLIP': (clip, {'lb' : 0}, {}),
            'RESCALEDICOM':(self._rescale_slope_intercept, 
                            {}, {}),
            'RESCALEGLOBAL': (self._undo_global_scale_fn, 
                                {}, {}),
        }
        return function_keys

    def __init__(self, task: object, model_dir: str, decrypt_key_hex: Optional[str] = None):
        """Initialize the job type object and initialize all the required variables

        :param task: subtle.dcmutils.dicom_filter.Task object of the task to execute
        :param model_dir: directory of the model to load for this job
        :param decrypt_key_hex: the decrypt key hexadecimal
        """
        self.model_dir = model_dir
        self.decrypt_key_hex = decrypt_key_hex

        BaseJobType.__init__(self, task, self.model_dir)

        self.task = task

        # define and prepare config
        proc_config = self.default_processing_config_gad.copy()
        proc_config.update(task.job_definition.exec_config)

        proc_config = self._parse_pipeline_definition(proc_config)

        k = list(proc_config.keys())
        _ = [proc_config.pop(f) for f in k if f not in self.SubtleGADProcConfig._fields]
        self._proc_config = self.SubtleGADProcConfig(**proc_config)
        

        # initialize arguments to update during processing
        # tuple of dict of input datasets by frame
        # (keys = frame sequence name, values = list of pydicom datasets)
        self._input_datasets = ()  # reference dataset list

        self._input_series = ()

        #itk_data
        self._itk_data = {}

        # dict of input arrays by frame
        # (keys = frame sequence name, values = pixel array)
        self._raw_input = {}

        # list of lambda functions constructed during preprocess
        self._proc_lambdas = []
        self.total_time = []

        self._output_data = {}
        self._undo_model_compat_reshape = None

        self._original_input_shape = None
        self._resampled_isotropic = False 
    
    @staticmethod
    def update_param_for_func(param, func):
        t = signature(func)
        param_func = param.copy()
        list_params_to_pop = [(k, k not in t.parameters) for k in param_func]
        _ = [param_func.pop(k) for k, to_pop in list_params_to_pop if to_pop]
        return param_func
    
    def _parse_pipeline_definition(self, proc_config):
        """
        Update the pipeline definition in the config
        :param proc_config:
        :return:
        """
        for pipeline_def in ["pipeline_preproc", "pipeline_postproc"]:
            #use the correct default pipeline based on model_type if it hasn't been updated
            if proc_config["model_type"] in proc_config[pipeline_def]:
                proc_config[pipeline_def] = proc_config[pipeline_def][proc_config["model_type"]]
            
            #make sure all the keys are step definitions
            if not all([re.search('STEP\d*', k) for k in proc_config[pipeline_def].keys()]):
                raise ValueError("Wrong keys found in pipeline definition ({}): {}".format(pipeline_def,
                                                                          proc_config[pipeline_def]))

            # make sure consecutive steps are defined
            lstep_numbers = [int(k.split('STEP')[1]) for k in proc_config[pipeline_def].keys()]

            lstep_numbers.sort()

            # update each step
            for step in proc_config[pipeline_def]:
                dict_step = proc_config[pipeline_def][step]
                if not 'op' in dict_step:
                    ("Operation key is missing in {} for {} definition: {}".format(step,
                                                                                  pipeline_def,
                                                                                  dict_step))

                # get function and default param based on operation keyword
                func, default_param, param_to_param = self.function_keys[dict_step['op']]

                if 'param' not in dict_step:
                    # use default param
                    dict_step['param'] = default_param
                    updated_param = False
                else:
                    # update default param with param defined in config
                    param = default_param.copy()
                    param.update(dict_step['param'])
                    dict_step['param'] = param
                    updated_param = True

                # Update pipeline param from config
                if param_to_param:
                    for config_key, param_key in param_to_param.items():
                        param_defined = config_key in proc_config and param_key in dict_step['param']
                        if param_defined and not updated_param:
                            dict_step['param'].update({config_key: proc_config[config_key]})

                # update step dictionary
                proc_config[pipeline_def][step] = dict_step

        # step specific validations
        # zoom must be in both pre- and post-processig or not at all
        # and it must have the same parameters in pre- and post-processing
        for op_name in ['SCALEGLOBAL']:
            op_in_preproc, step_num_pre, step_pre = self._is_step_in_pipeline(op_name, proc_config["pipeline_preproc"])
            op_in_postproc, step_num_post, step_post = self._is_step_in_pipeline('RE'+op_name, proc_config["pipeline_postproc"])

            # Make sure operation is present in both pre- and post-processing OR absent from both
            if op_in_preproc != op_in_postproc:
               raise ValueError("{} operation was present in pre-processing but not in post-processing or vice and versa.".format(op_name))


        return proc_config

    @staticmethod
    def _is_step_in_pipeline(step_op_name: str, pipeline_dict: dict):
        step_num = ''
        step_dict = {}
        found_step = False

        for step_num, step_dict in pipeline_dict.items():
            if step_dict['op'] == step_op_name:
                # Step was found
                found_step = True
                break

        if not found_step:
            step_num = ''
            step_dict = {}

        return found_step, step_num, step_dict

    def _get_step_by_name(self, step_op_name: str):
        # Try finding step in pre-processing
        op_in_preproc, _, step_dict = self._is_step_in_pipeline(step_op_name,
                                                                self._proc_config.pipeline_preproc)

        # if not found, try in post-processing
        if not op_in_preproc:
            _, _, step_dict = self._is_step_in_pipeline(step_op_name,
                                                        self._proc_config.pipeline_postproc)

        return step_dict

    def _preprocess_pixel_data(self):
        """
        get pixel data from dicom datasets
        and resample the pixel data to the specified shape
        :return: preprocessed pixel data
        """
        # apply preprocessing for each frame of the dictionary
        
        dict_input_data = {}
        #print(self._proc_config.pipeline_preproc)
        for frame_seq_name, data_array in self._raw_input.items():
            # check the shape, if it is not square, pad to square
            # because the saved model need a fixed receptive field
            input_data= data_array.copy()  # better to be deep.copy()
            #assert len(input_data.shape) == 3, "ERROR: data is not 3D"

            for i in range(1, len(self._proc_config.pipeline_preproc)+1):
                # get the step info
                step_dict = self._proc_config.pipeline_preproc['STEP{}'.format(i)]
                # get the function to run for this step
                func = self.function_keys[step_dict['op']][0]
                # run function
                if step_dict['op'] == "MASK":
                    input_data, mask = func(input_data, step_dict['param'])
                    raw_input_images = np.copy(input_data)
                    self.mask = np.copy(mask)
                else:
                    input_data, raw_input_images = func(input_data, raw_input_images, step_dict['param'])

            # assign value per frame
            (raw_input_images, self._undo_model_compat_reshape) = \
            self._process_model_input_compatibility(raw_input_images)
            dict_input_data[frame_seq_name] = raw_input_images

        #np.save('/home/SubtleGad/app/tests/data/philips_preprocess.npy', raw_input_images)
        

        return dict_input_data

    def _preprocess(self) -> Dict:
        """
        Preprocess function called by the __call__ method of BaseJobType. Before calling the actual
        preprocessing logic, this method calls a bunch of initialization functions to get DICOM
        data, set manufacturer specific config, get the DICOM pixel spacing, get the DICOM scaling
        information and finally get the raw pixel data

        """
        t1 = time.time()
        self._get_input_series()

        self._get_pixel_spacing()
        self._get_dicom_scale_coeff()

        # get original pixel data and meta data info
        self._get_raw_pixel_data()

        dict_pixel_data = self._preprocess_pixel_data()

        t2 = time.time()
        total_time['preprocessing'] = t2 -t1

        print('preprocessing_time', total_time['preprocessing'])

        total_time['preprocess_end_time'] = t2

        # dictionary of the data to process by frame
        #dict_pixel_data = self._preprocess_raw_pixel_data()
        return dict_pixel_data

    def _metadata_compatibility(self):

        if self._proc_config.metadata_comp:
            
            for key in self._proc_config.metadata_comp.keys():

                if key == 'rows':

                    if not self._inputs['zd'].rows or not self._inputs['ld'].rows:
                        raise ValueError(f'{key} metadata not found in the zd or ld series dicom')
                    else:
                        if not np.isclose(self._inputs['zd'].rows , self._inputs['ld'].rows, atol = self._proc_config.metadata_comp[key]):

                            raise ValueError(f'{key} is not compatible with the given tolerance level of {self._proc_config.metadata_comp[key]}')
                        else:
                            self._logger.info(f'{key} metadata compatibility passed')

                if key == 'columns':

                    if not self._inputs['zd'].columns or not self._inputs['ld'].columns:
                        raise ValueError(f'{key} metadata not found in the zd or ld series dicom')
                    else:
                        if not np.isclose(self._inputs['zd'].columns , self._inputs['ld'].columns, atol = self._proc_config.metadata_comp[key]):

                            raise ValueError(f'{key} is not compatible with the given tolerance level of {self._proc_config.metadata_comp[key]}')
                        else:
                            self._logger.info(f'{key} metadata compatibility passed')

                if key == 'pixelspacing':

                    if not self._inputs['zd'].pixelspacing or not self._inputs['ld'].pixelspacing:
                        raise ValueError(f'{key} metadata not found in the zd or ld series dicom')
                    else:
                        if not np.isclose(self._inputs['zd'].pixelspacing[0] , self._inputs['ld'].pixelspacing[0], atol = self._proc_config.metadata_comp[key]):

                            raise ValueError(f'{key} is not compatible with the given tolerance level of {self._proc_config.metadata_comp[key]}')
                        else:
                            self._logger.info(f'{key} metadata compatibility passed')

                if key ==  'patientid':
                    
                    if not self._inputs['zd'].patientid or not self._inputs['ld'].patientid:
                        raise ValueError(f'{key} metadata not found in the zd or ld series dicom')
                    else:
                        if not np.isclose(self._inputs['zd'].patientid , self._inputs['ld'].patientid, atol = self._proc_config.metadata_comp[key]):
                            raise ValueError(f'{key} is not compatible with the given tolerance level of {self._proc_config.metadata_comp[key]}')
                        else:
                            self._logger.info(f'{key} metadata compatibility passed')
                
                if key == 'imagepositionpatient':

                    if not self._inputs['zd'].imagepositionpatient or not self._inputs['ld'].imagepositionpatient:
                        raise ValueError(f'{key} metadata not found in the zd or ld series dicom')
                    else:
                        if not np.all(np.isclose(self._inputs['zd'].imagepositionpatient , self._inputs['ld'].imagepositionpatient, atol = self._proc_config.metadata_comp[key])):
                            raise ValueError(f'{key} is not compatible with the given tolerance level of {self._proc_config.metadata_comp[key]}')
                        else:
                            self._logger.info(f'{key} metadata compatibility passed')

                if key == 'slicethickness':

                    if not self._inputs['zd'].slicethickness or not self._inputs['ld'].slicethickness:
                        raise ValueError(f'{key} metadata not found in the zd or ld series dicom')
                    else:
                        if not np.isclose(self._inputs['zd'].slicethickness , self._inputs['ld'].slicethickness, atol = self._proc_config.metadata_comp[key]):
                            raise ValueError(f'{key} is not compatible with the given tolerance level of {self._proc_config.metadata_comp[key]}')
                        else:
                            self._logger.info(f'{key} metadata compatibility passed')

                if key == 'accessionnumber':

                    if not self._inputs['zd'].accessionnumber or not self._inputs['ld'].accessionnumber:
                        raise ValueError(f'{key} metadata not found in the zd or ld series dicom')
                    else:
                        if self._inputs['zd'].accessionnumber != self._inputs['ld'].accessionnumber:
                            raise ValueError(f'{key} is not compatible with the given tolerance level of {self._proc_config.metadata_comp[key]}')
                        else:
                            self._logger.info(f'{key} metadata compatibility passed')

    def _get_available_gpus(self) -> str:
        """
        Return GPUs indices that have at least the minimum GPU memory required for processing

        :return: Comma separated values of GPUs available based on the min_gpu_mem_mb configuration
        """
        stats = GPUtil.getGPUs()
        return ','.join([
            str(gpu.id) for gpu in stats
            if (gpu.memoryTotal - gpu.memoryUsed) >= self._proc_config.min_gpu_mem_mb
        ])

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-locals
    @processify
    def _process(self, dict_pixel_data: Dict) -> Dict:
        """
        Take the pixel data and launch a Pool of processes to do MPR processing in parallel. Once
        the parallel inference jobs are over, average the results and return the output data.
        This method is called by the __call__ method of BaseJobType

        This method has a @processify decorator which makes it run as a separate process - this is
        required multiple jobs are run sequentially. Once a single job completes, the decorator
        will make the process exit, which will free up the GPU memory.

        :param dict_pixel_data: dictionary of the input pixel data (numpy arrays) by frame
        :return: the processed output data
        """
        t1 = time.time()
        dict_output_array = {}

        self._logger.info("starting inference (job type: %s)", self.name)

        if self._proc_config.allocate_available_gpus:
            gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", self._get_available_gpus())
        else:
            gpu_devices = "0"

        avail_gpu_ids = gpu_devices.split(',')

        self._logger.info("avail gpu ids %s", avail_gpu_ids)

        if not avail_gpu_ids:
            msg = "Adequate computing resources not available at this moment, to complete the job"
            self._logger.error(msg)
            raise Exception(msg)

        gpu_repeat = [[id] * self._proc_config.num_procs_per_gpu for id in avail_gpu_ids]
        gpu_ids = [item for sublist in gpu_repeat for item in sublist]

        gpu_q = Queue()
        for gid in gpu_ids:
            gpu_q.put(gid)

        process_pool = Pool(processes=len(gpu_ids), initializer=_init_gpu_pool, initargs=(gpu_q, ))

        model_input_shape = (
                1, dict_pixel_data['single_frame'].shape[2], dict_pixel_data['single_frame'].shape[3], 2 * self._proc_config.slices_per_input
            )

        model = GenericInferenceModel(
                model_dir=self.model_dir, decrypt_key_hex=self.decrypt_key_hex,
                input_shape=model_input_shape
            )
            #model._model_obj().eval()

        model.update_config(self.task.job_definition.exec_config)

        for frame_seq_name, pixel_data in dict_pixel_data.items():
            # perform inference with default input format (NHWC)

            angle_start = self._proc_config.mpr_angle_start
            angle_end = self._proc_config.mpr_angle_end
            num_rotations = self._proc_config.num_rotations

            mpr_params = []

            _, n_slices, height, width = pixel_data.shape
            #print('prev pixel data shape', pixel_data.shape)

            param_obj = {
                'model_dir': self.model_dir,
                'decrypt_key_hex': self.decrypt_key_hex,
                'exec_config': self.task.job_definition.exec_config,
                'slices_per_input': self._proc_config.slices_per_input,
                'reshape_for_mpr_rotate': self._proc_config.reshape_for_mpr_rotate,
                #'inference_mpr': self._proc_config.inference_mpr,
                'num_rotations': self.task.job_definition.exec_config['num_rotations'],
                'data': (pixel_data[:,:,:,:])
            }

            #print('pob datas', pixel_data.shape)

            if param_obj['exec_config']['inference_mpr']:
                slice_axes = [0, 2, 3]
            else:
                slice_axes = [0]

            if param_obj['exec_config']['inference_mpr'] and param_obj['exec_config']['num_rotations'] > 1:
                angles = np.linspace(0, 90, param_obj['exec_config']['num_rotations'], endpoint=False)
            else:
                angles = [0]

            #slice_axes = np.arange(1, 4) if not self._proc_config.skip_mpr else np.array([1])
            predictions = []
            for angle in angles:#np.linspace(angle_start, angle_end, num=num_rotations, endpoint=False):
                for slice_axis in slice_axes:
                    pobj = copy.deepcopy(param_obj)
                    pobj['angle'] = angle
                    pobj['slice_axis'] = slice_axis
                    mpr_params.append(pobj)
                    predictions.append(SubtleGADJobType._process_mpr(pobj, model))

            #predictions = SubtleGADJobType._process_mpr(mpr_params)
            # Convert the array to a shape of (n_slices, height, width, 3, num_rotations)
            _reshape_tuple = (len(slice_axes), num_rotations, n_slices, height, width, 1)
            predictions = np.array(predictions).reshape(_reshape_tuple)
            #print('predictions shape', predictions.shape)
            tx_order = (2, 3, 4, 1, 0)
            predictions = np.array(predictions)[..., 0].transpose(tx_order)
            #print(predictions)

            # compute a volume of shape (n_slices, height, width) which has 1 for all the pixels
            # with pixel value > 0 and 0 otherwise - across all mpr predictions
            mask_sum = np.sum(
                np.array(predictions > 0, dtype=np.float), axis=(-1, -2), keepdims=False
            )

            # now average the mpr inferences using the non-zero mask sum
            output_array = np.divide(
                np.sum(predictions, axis=(-1, -2), keepdims=False),
                mask_sum, where=(mask_sum > 0)
            )[..., None]
            #print(output_array)
            #print(output_array.shape)
            # undo zero padding and isotropic resampling
            output_array = self._undo_reshape(output_array)

            dict_output_array[frame_seq_name] = output_array

        self._logger.info("inference finished")

        t2 = time.time()

        total_time['model_inference'] = t2 - t1

        return dict_output_array

    # pylint: disable=arguments-differ
    def _postprocess(self, dict_pixel_data: Dict, out_dicom_dir: str):
        """
        Undo some of the preprocessing (scaling) steps for the DICOM pixel data to match that
        of the input data. This method is called by the __call__ method of the BaseJobType

        :param dict_pixel_data: dict of the result pixel data in a numpy array by frame.
        :param out_dicom_dir: the directory or tarball of output DICOM files
        """

        t1 = time.time()

        total_time['model_inference'] =  t1 - total_time['preprocess_end_time'] 

        total_time.pop('preprocess_end_time')
        # check input size
        for frame_seq_name in self._input_datasets[0]:
            self._logger.info("postprocess shape %s", dict_pixel_data[frame_seq_name].shape)

            if (
                    len(self._input_datasets[0][frame_seq_name])
                    != dict_pixel_data[frame_seq_name].shape[0]
            ):
                msg = "postprocess() got mismatched pixel data and template DICOM files"
                self._logger.error(msg)
                raise ValueError(msg)

        # data post processing
        self._postprocess_data(dict_pixel_data)
        # data saving
        t2 = time.time()

        total_time['post_process'] = t2 - t1

        self._save_data(self._output_data, self._input_datasets[0], out_dicom_dir)

        t3 = time.time()

        total_time['save_data'] = t3 - t2

    def _get_input_series(self):
        """
        Get dicom datasets from the DicomSeries passed through the task as a dictionary of sorted
        datasets by frame: keys are frame sequence name and values are lists of pydicom datasets
        """

        if len(self.task.dict_required_series) != 2:
            raise KeyError("More than two input series, only zero-dose/ low-dose are allowed")

        self._input_series = list(self.task.dict_required_series.values())

        #Assert both zero-dose and low-dose are DicomSeries Objects
        if not isinstance(self._input_series[0], series_utils.DicomSeries) and \
                not isinstance(self._input_series[1], series_utils.DicomSeries):
                    raise TypeError("Input series should be a DicomSeries or IsmrmrdSeries object")

        zero_dose_series = [(series) for series in self._input_series  if 'ZERO_DOSE' in series.seriesdescription.upper()][0]
        low_dose_series =  [(series) for series in self._input_series  if 'LOW_DOSE' in series.seriesdescription.upper()][0]

        self._inputs = {'zd' : zero_dose_series, 'ld': low_dose_series}

        #Check for one zero dose, low dose
        zero_dose_pixels =[list(zero_dose_series.get_pixel_data(rescale=False).values())][0][0]
        low_dose_pixels = [list(low_dose_series.get_pixel_data(rescale=False).values())][0][0]
        #print('low_dose_pixels', low_dose_pixels.shape)

        self._input_datasets = (zero_dose_series.get_dict_sorted_datasets(),
                                low_dose_series.get_dict_sorted_datasets()
        )
        
        try:
            
            self._itk_data['zero_dose'] = zero_dose_series._list_itks[0]
            self._itk_data['low_dose'] = low_dose_series._list_itks[0]
        except:
            
            self._itk_data['zero_dose'] = sitk.GetImageFromArray(zero_dose_pixels)
            self._itk_data['low_dose'] = sitk.GetImageFromArray(low_dose_pixels)
            #print(self._itk_data['zero_dose'])

    def _get_raw_pixel_data(self):
        """
        Read the input series and set raw pixel data as a numpy array
        """
        self._metadata_compatibility()

        for frame_seq_name in self._input_datasets[0].keys():
            zero_data_np = self._input_series[0].get_pixel_data()[frame_seq_name]
            low_data_np = self._input_series[1].get_pixel_data()[frame_seq_name]

            self._raw_input[frame_seq_name] = np.array([zero_data_np, low_data_np])
            self._logger.info("the shape of array %s", self._raw_input[frame_seq_name].shape)

    def _mask_noise_process(self, images: np.ndarray, param: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform noise masking which removes noise around the brain caused due to interference,
        eye blinking etc.
        :param images: Input zero and low dose images as a numpy array
        :return: A tuple of the masked image and the binary mask computed
        """
        mask = []
        self._logger.info("Performing noise masking...")

        threshold = param['noise_mask_threshold']
        mask_area = param['noise_mask_area']
        use_selem = param['noise_mask_selem']

        mask_fn = lambda idx: (lambda ims: ims[idx] * \
        preprocess_single_series.mask_bg_noise(ims[idx], threshold, mask_area, use_selem))

        for idx in range(images.shape[0]):
            noise_mask = preprocess_single_series.mask_bg_noise(
                img=images[idx], threshold=threshold, noise_mask_area=mask_area,
                use_selem=use_selem
            )
            images[idx, ...] *= noise_mask
            mask.append(noise_mask)

        self._proc_lambdas.append({
            'name': 'noise_mask',
            'fn': [mask_fn(0), mask_fn(1)]
        })

        return images, np.array(mask)

    def _mask_noise(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform noise masking which removes noise around the brain caused due to interference,
        eye blinking etc.
        :param images: Input zero and low dose images as a numpy array
        :return: A tuple of the masked image and the binary mask computed
        """
        mask = []

        if self._proc_config.perform_noise_mask:
            self._logger.info("Performing noise masking...")

            threshold = self._proc_config.noise_mask_threshold
            mask_area = self._proc_config.noise_mask_area
            use_selem = self._proc_config.noise_mask_selem

            mask_fn = lambda idx: (lambda ims: ims[idx] * \
            preprocess_single_series.mask_bg_noise(ims[idx], threshold, mask_area, use_selem))

            for idx in range(images.shape[0]):
                noise_mask = preprocess_single_series.mask_bg_noise(
                    img=images[idx], threshold=threshold, noise_mask_area=mask_area,
                    use_selem=use_selem
                )
                images[idx, ...] *= noise_mask
                mask.append(noise_mask)

            self._proc_lambdas.append({
                'name': 'noise_mask',
                'fn': [mask_fn(0), mask_fn(1)]
            })

        return images, np.array(mask)

    #@processify
    def apply_brain_mask(self, ims, raw_input_images , param: dict):

        brain_mask = self._brain_mask(ims)

        ims_mask = np.copy(ims)

        if True:
            #print('Applying computed brain masks on images. Mask shape -', brain_mask.shape)
            ims_mask = np.zeros_like(ims)

            if brain_mask.ndim == 4:
                ims_mask = brain_mask * ims
            else:
                for cont in range(ims.shape[0]):
                    ims_mask[cont,:, :, :] = brain_mask * ims[cont,:, :, :]

        return ims_mask, raw_input_images

    def _has_dicom_scaling_info(self) -> bool:
        """
        Use the DICOM header of the input dataset to find whether scaling information is available
        :return: Boolean indicating whether scaling information is available
        """
        header_zero, _ = self._get_dicom_header()
        return 'RescaleSlope' in header_zero

    def _get_dicom_header(self) -> Tuple[object, object]:
        """
        Get the DICOM header (pydicom.dataset) of the zero and low dose images
        :return: Tuple of pydicom dataset objects of zero and low dose images
        """
        for frame_seq_name, _ in self._input_datasets[0].items():
            header_zero = self._input_datasets[0][frame_seq_name][0]
            header_low = self._input_datasets[1][frame_seq_name][0]

            return (header_zero, header_low)

    def _get_dicom_scale_coeff(self):
        """
        Sets the scale coefficient attribute with the scaling information present in the DICOM
        header. If scaling information is not available, this method returns the default values.
        """
        #print('does dicom have scaling info ', self._has_dicom_scaling_info())
        if not self._has_dicom_scaling_info():
            self._dicom_scale_coeff = [{
                'rescale_slope': 1.0,
                'rescale_intercept': 0.0,
                'scale_slope': 1.0
            }, {
                'rescale_slope': 1.0,
                'rescale_intercept': 0.0,
                'scale_slope': 1.0
            }]

        else:
            header_zero, header_low = self._get_dicom_header()
            self._dicom_scale_coeff = [{
                'rescale_slope': float(header_zero.RescaleSlope),
                'rescale_intercept': float(header_zero.RescaleIntercept),
                'scale_slope': float(header_zero[0x2005, 0x100e].value)
            }, {
                'rescale_slope': float(header_low.RescaleSlope),
                'rescale_intercept': float(header_low.RescaleIntercept),
                'scale_slope': float(header_low[0x2005, 0x100e].value)
            }]

    @staticmethod
    def _scale_slope_intercept(
        img: np.ndarray, rescale_slope: float, rescale_intercept: float, scale_slope: float
    ) -> np.ndarray:
        """
        Static method to scale the input image with the given rescale slope and intercept and scale
        slope

        :param img: Input image as numpy array
        :param rescale_slope: Slope part of the linear transformation of pixels from disk
        representation to memory representation
        :param rescale_intercept: Intercept part of the linear transformation of pixels from disk
        representation to memory representation
        :param scale_slope: Private philips tag for scale slope information
        :return: Scaled image as numpy array
        """

        return (img * rescale_slope + rescale_intercept) / (rescale_slope * scale_slope)

    @staticmethod
    def _rescale_slope_intercept(
        img: np.ndarray) -> np.ndarray:
        """
        Static method to undo the dicom scaling

        :param img: Input image as numpy array
        :param rescale_slope: Slope part of the linear transformation of pixels from disk
        representation to memory representation
        :param rescale_intercept: Intercept part of the linear transformation of pixels from disk
        representation to memory representation
        :param scale_slope: Private philips tag for scale slope information
        :return: Rescaled image as numpy array
        """
        if self._dicom_scale_coeff:
            rescale_slope = self._dicom_scale_coeff[0]['rescale_slope']
            scale_slope = self._dicom_scale_coeff[0]['scale_slope']
            rescale_intercept = self._dicom_scale_coeff[0]['rescale_intercept']

        return (img * rescale_slope * scale_slope - rescale_intercept) / rescale_slope

    def _scale_intensity_with_dicom_tags(self, images: np.ndarray, raw_input_images: np.ndarray, param:dict) -> np.ndarray:
        """
        Helper function to scale the image intensity using DICOM header information

        :param images: Zero and low dose images as numpy array
        :return: The scaled images as numpy array
        """
        #if self._proc_config.perform_dicom_scaling and self._has_dicom_scaling_info():
        self._logger.info("Performing intensity scaling using DICOM tags...")

        scaled_images = np.copy(images)
        raw_scaled_images = np.copy(raw_input_images)

        scale_fn = lambda idx: (
            lambda ims: SubtleGADJobType._scale_slope_intercept(ims[idx],
            **self._dicom_scale_coeff[idx])
        )

        scaled_images[0] = scale_fn(0)(scaled_images)
        scaled_images[1] = scale_fn(1)(scaled_images)

        raw_scaled_images[0] = scale_fn(0)(raw_scaled_images)
        raw_scaled_images[1] = scale_fn(1)(raw_scaled_images)

        self._proc_lambdas.append({
            'name': 'dicom_scaling',
            'fn': [scale_fn(0), scale_fn(1)]
        })

        #np.save(f'./input/IM001//expected_dicom_scaling.npy', raw_scaled_images)

        self._undo_dicom_scale_fn = lambda img: SubtleGADJobType._rescale_slope_intercept(img,
        **self._dicom_scale_coeff[0])

        return scaled_images, raw_scaled_images


    def _get_pixel_spacing(self):
        """
        Obtains the pixel spacing information from DICOM header of input dataset and sets it to the
        local pixel spacing attribute
        """
        #header_zero, header_low = self._get_dicom_header()

        x_zero, y_zero = self._inputs['zd'].pixelspacing
        z_zero = self._inputs['zd'].slicethickness

        x_low, y_low = self._inputs['ld'].pixelspacing
        z_low = self._inputs['ld'].slicethickness

        self._pixel_spacing = [(x_zero, y_zero, z_zero), (x_low, y_low, z_low)]

    def _register(self, images: np.ndarray, raw_input_images: np.ndarray, param: dict) -> np.ndarray:
        """
        Performs image registration of low dose image by having the zero dose as the reference
        :param images: Input zero dose and low dose images as numpy array
        :return: Registered images as numpy array
        """
        """
        Performs image registration of low dose image by having the zero dose as the reference
        :param images: Input zero dose and low dose images as numpy array
        :return: Registered images as numpy array
        """
        reg_images = np.copy(images)
        raw_reg_images = np.copy(raw_input_images)

        # if not self._proc_config.perform_registration:
        #     self._logger.info("Skipping image registration...")
        #     return reg_images

        self._logger.info("Performing registration of low dose with respect to zero dose...")

        # register the low dose image with respect to the zero dose image
        reg_args = {
            'fixed_spacing': self._pixel_spacing[0],
            'moving_spacing': self._pixel_spacing[1],
            'transform_type': param["transform_type"]
        }
        reg_images[1], reg_params = preprocess_single_series.register(
            fixed_img=reg_images[0], moving_img=reg_images[1], **reg_args
        )

        idty_fn = lambda ims: ims[0]

        if param["use_mask_reg"]:
            apply_reg = lambda ims: preprocess_single_series.apply_registration(ims[1], \
            self._pixel_spacing[1], reg_params)
        else:
            self._logger.info('Registration will be re-run for unmasked images...')
            apply_reg = lambda ims: preprocess_single_series.register(
                fixed_img=ims[0], moving_img=ims[1], return_params=False, **reg_args
            )
        
        raw_reg_images[1] = apply_reg(raw_reg_images)

        self._proc_lambdas.append({
            'name': 'register',
            'fn': [idty_fn, apply_reg]
        })

        #np.save(f'./input/IM001//expected_register.npy', raw_reg_images)

        return reg_images, raw_reg_images
    

    def _match_histogram(self, images: np.ndarray, raw_input_images: np.ndarray, param: dict) -> np.ndarray:
        """
        Performs histogram matching of low dose by having zero dose as reference
        :param images: Input zero dose and low dose images as numpy array
        :return: Histogram matched images as numpy array
        """
        self._logger.info("Matching histogram of low dose with respect to zero dose...")
        hist_images = np.copy(images)
        raw_hist_images = np.copy(raw_input_images)

        hist_images[1] = preprocess_single_series.match_histogram(
            img=hist_images[1], ref_img=hist_images[0]
        )

        idty_fn = lambda ims: ims[0]
        hist_match_fn = lambda ims: preprocess_single_series.match_histogram(ims[1], ims[0])

        raw_hist_images[1] = hist_match_fn(raw_hist_images)

        self._proc_lambdas.append({
            'name': 'histogram_matching',
            'fn': [idty_fn, hist_match_fn]
        })

        #np.save(f'./input/IM001//expected_histogram_matching.npy',self.raw_input_images)
        return hist_images, raw_hist_images

    def _scale_intensity(self, images: np.ndarray, raw_input_images: np.ndarray, param: dict) -> np.ndarray:
        """
        Performs relative and global scaling of images by using the pixel values inside the
        noise mask previously computed

        :param images: Input zero dose and low dose images as numpy array
        :param noise_mask: Binary noise mask computed in an earlier step of pre-processing
        :return: Scaled images as numpy array
        """
        scale_images = np.copy(images)
        raw_scale_images = np.copy(raw_input_images)

        self._logger.info("Performing intensity scaling...")
        num_slices = scale_images.shape[1]

        idx_center = range(
            num_slices//2 - param["num_scale_context_slices"]//2,
            num_slices//2 + param["num_scale_context_slices"]//2
        )

        # use pixels inside the noise mask of zero dose
        #print('shape of noise mask ', noise_mask.shape)
        #print('shape of scale images ', scale_images.shape)
        
        ref_mask = self.mask[0, idx_center]

        context_img_zero = scale_images[0, idx_center, ...][ref_mask != 0].ravel()
        context_img_low = scale_images[1, idx_center, ...][ref_mask != 0].ravel()

        context_img = np.stack((context_img_zero, context_img_low), axis=0)

        scale_factor = preprocess_single_series.get_intensity_scale(
            img=context_img[1], ref_img=context_img[0], levels=np.linspace(.5, 1.5, 30)
        )

        self._logger.info("Computed intensity scale factor is %s", scale_factor)

        scale_images[1] *= scale_factor
        context_img[1] *= scale_factor

        sfactors = [1, scale_factor]
        match_scales_fn = lambda idx: (lambda ims: ims[idx] * sfactors[idx])

        raw_scale_images[0] = match_scales_fn(0)(raw_scale_images)
        raw_scale_images[1] = match_scales_fn(1)(raw_scale_images)

        self._proc_lambdas.append({
            'name': 'match_scales',
            'fn': [match_scales_fn(0), match_scales_fn(1)]
        })

        #np.save(f'./input/IM001//expected_match_scales.npy',raw_scale_images)

        #raw_global_image = np.copy(raw_scale_images)

        context_img = context_img.transpose(1, 0)
        scale_images = scale_images.transpose(1, 0, 2, 3)

        norm_axis = (0, 1) if param["joint_normalize"] else (0, )

        if param["scale_ref_zero_img"]:
            norm_img = context_img[..., 0]
            norm_axis = (0, )
        else:
            norm_img = context_img

        norm_img[norm_img < 0] = 0
        global_scale = np.mean(norm_img, axis=norm_axis)
        for a in norm_axis:
            global_scale = np.expand_dims(global_scale, axis=a)

        # repeat the computed scale such that its shape is always (1, 2)
        if global_scale.ndim == 1:
            global_scale = np.repeat([global_scale], repeats=scale_images.shape[1], axis=1)
        elif global_scale.ndim == 2 and global_scale.shape[1] == 1:
            global_scale = np.repeat(global_scale, repeats=scale_images.shape[1], axis=1)

        self.global_scale = global_scale[:, :, None, None]
        self._logger.info("Computed global scale %s with shape %s", global_scale, \
        global_scale.shape)

        scale_images /= self.global_scale
        global_scale_fn = lambda idx: (
            lambda ims: (ims[idx] / self.global_scale[:, idx])
        )

        raw_scale_images[0] = global_scale_fn(0)(raw_scale_images)
        raw_scale_images[1] = global_scale_fn(1)(raw_scale_images)

        #np.save(f'./input/IM001//expected_global_scale.npy',raw_scale_images)

        #self._undo_global_scale_fn = lambda img: img * global_scale[:, 0]
        # self._proc_lambdas.append({
        #     'name': 'global_scale',
        #     'fn': [global_scale_fn(0), global_scale_fn(1)]
        # })

        return scale_images.transpose(1, 0, 2, 3), raw_scale_images

    def _undo_global_scale_fn(self,img, param):

        return img*self.global_scale[:,0]

    #@processify
    def _process_model_input_compatibility(self, input_data: np.ndarray) -> np.ndarray:
        """
        Check if input data is compatible with the model input shape. If not resample
        input to isotropic resolution and zero pad to make the input work with the given model.

        :param input_data: Input data numpy array
        :return: Resampled/zero-padded data
        """
        # Model is initialized here only to get the default input shape
        undo_methods = []
        orig_shape = input_data.shape
        #print('what is model_dir', self.model_dir)
        #model = GenericInferenceModel(
        #    model_dir=self.model_dir, decrypt_key_hex=self.decrypt_key_hex
        #)
        #print(model._model_obj)
        #model.update_config(self.task.job_definition.exec_config)
        model_input = [14,512,512]#model._model_obj.inputs[0]
        #print('shape',model._model_obj.inputs[0])
        model_shape = (int(model_input[1]), int(model_input[2]))
        input_shape = (input_data.shape[-2], input_data.shape[-1])

        if input_shape != model_shape:
            if input_shape[0] != input_shape[1]:
                max_shape = np.max([input_shape[0], input_shape[1]])
                self._logger.info('Zero padding to %s %s', max_shape, max_shape)
                input_data = SubtleGADJobType._zero_pad(
                    img=input_data,
                    ref_img=np.zeros(
                        (input_data.shape[0], input_data.shape[1], max_shape, max_shape)
                    )
                )

                undo_methods.append({
                    'fn': 'undo_zero_pad',
                    'arg': np.zeros(orig_shape[1:])
                })

            pixel_spacing = np.round(np.array(self._pixel_spacing[0])[::-1], decimals=2)

            if not np.array_equal(pixel_spacing, self._proc_config.model_resolution):
                self._logger.info('Resampling to isotropic resolution...')

                resize_factor = (pixel_spacing / self._proc_config.model_resolution)
                new_shape = np.round(input_data[0].shape * resize_factor)
                real_resize_factor = new_shape / input_data[0].shape
                #print('resizing factor', resize_factor)
                zero_resample = zoom_interp(input_data[0], resize_factor)
                low_resample = zoom_interp(input_data[1], resize_factor)

                input_data = np.array([zero_resample, low_resample])

                undo_factor = 1.0 / resize_factor
                undo_methods.append({
                    'fn': 'undo_resample',
                    'arg': undo_factor
                })

                self._logger.info('Input data shape after resampling %s', input_data.shape)

            if (input_data.shape[-2], input_data.shape[-1]) != model_shape:
                self._logger.info('Zero padding to fit model shape...')
                curr_shape = input_data.shape
                input_data = SubtleGADJobType._zero_pad(
                    img=input_data,
                    ref_img=np.zeros(
                        (input_data.shape[0], input_data.shape[1], model_shape[0], model_shape[1])
                    )
                )

                undo_methods.append({
                    'fn': 'undo_zero_pad',
                    'arg': np.zeros(curr_shape[1:])
                })

                self._logger.info('Input data shape after zero padding %s', input_data.shape)

        return input_data, undo_methods

    @staticmethod
    def _get_crop_range(shape_num: int) -> Tuple[int, int]:
        """
        Gets starting and ending range of indices for a given shape number. Helps in center
        cropping an image

        :param shape_num: The shape number for which the range needs to be computed
        :return: Computed range with start and end indices
        """
        if shape_num % 2 == 0:
            start = end = shape_num // 2
        else:
            start = (shape_num + 1) // 2
            end = (shape_num - 1) // 2

        return (start, end)

    @staticmethod
    def _center_crop(img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
        """
        Performs center cropping of a given image, to the shape of the given ref_img
        :param img: Numpy array of the input image which needs to be center cropped
        :param ref_img: The input img is center cropped to resemble the shape of ref_img
        :return: Center cropped image as numpy array
        """
        s = []
        e = []

        for i, sh in enumerate(img.shape):
            if sh > ref_img.shape[i]:
                diff = sh - ref_img.shape[i]
                if diff == 1:
                    s.append(0)
                    e.append(sh-1)
                else:
                    start, end = SubtleGADJobType._get_crop_range(diff)
                    s.append(start)
                    e.append(-end)
            else:
                s.append(0)
                e.append(sh)

        return img[s[0]:e[0], s[1]:e[1], s[2]:e[2]]

    @staticmethod
    def _zero_pad(img: np.ndarray, ref_img: np.ndarray, const_val: int = 0) -> np.ndarray:
        """
        Opposite of center cropping, pads img with const_val (by default 0) to match the shape
        of ref_img
        :param img: Numy array of the input image which needs to be zero padded
        :param ref_img: The input img is zero padded to resemble the shape of ref_img
        :return: Zero padded image as numpy array
        """

        npad = []
        for i, sh in enumerate(ref_img.shape):
            if img.shape[i] >= sh:
                npad.append((0, 0))
            else:
                diff = (sh - img.shape[i]) // 2
                npad.append((diff, diff))

        return np.pad(img, pad_width=npad, mode='constant', constant_values=const_val)
    
    #@staticmethod
    #@processify
    def _mask_npy(self,img_npy, dcm_ref, hdbet):
        mask = None
        
        data_sitk = sitk.GetImageFromArray(img_npy)

        #print('size mask npy',data_sitk.GetSize())
        
        data_sitk.CopyInformation(dcm_ref)
        
        mask = hdbet.run_hd_bet_inmem(data_sitk)

        return mask

    #@staticmethod
    @processify
    def _brain_mask(self,ims):
        ims_proc = np.copy(ims)
        mask = None
        #self.hdbet = HDBetInMemory()

        if True:
            #print('Extracting brain regions using deepbrain...')
            device = int(0)
            ## Temporarily using DL based method for extraction
            hdbet = HDBetInMemory(mode = 'fast', model_path = self.model_dir, device =device, do_tta = False)

            mask_zero = self._mask_npy(ims_proc[0,:, ...], self._itk_data['zero_dose'], hdbet)

            if True:
                mask_low = self._mask_npy(ims_proc[1,:,  ...], self._itk_data['low_dose'], hdbet)
                #mask_full = _mask_npy(ims[:, 2, ...], self._itk_data['full_dose'], device)

                if False:
                    # union of all masks
                    mask = ((mask_zero > 0 ) | (mask_low > 0) | (mask_full > 0))
                else:
                    mask = np.array([mask_zero, mask_low]).transpose(0,1, 2, 3)
            else:
                mask = mask_zero

        return mask


    def _postprocess_data(self, dict_pixel_data: Dict):
        """
        Post process the pixel data by undoing certain scaling operations
        :param dict_pixel_data: dict of data to post process by frame
        """
        for frame_seq_name, data_array in dict_pixel_data.items():
            # check the shape, if it is not square, pad to square
            # because the saved model need a fixed receptive field
            input_data= data_array.copy()  # better to be deep.copy()
            #assert len(input_data.shape) == 3, "ERROR: data is not 3D"
            if len(input_data.shape) == 4:
                input_data = input_data[..., 0]

            for i in range(1, len(self._proc_config.pipeline_postproc)+1):
                # get the step info
                step_dict = self._proc_config.pipeline_postproc['STEP{}'.format(i)]
                # get the function to run for this step
                func = self.function_keys[step_dict['op']][0]
                # run function
                input_data= func(input_data, step_dict['param'])

            self._output_data[frame_seq_name] = input_data

        # for frame_seq_name, pixel_data in dict_pixel_data.items():
        #     # get 3D volume
        #     if len(pixel_data.shape) == 4:
        #         pixel_data = pixel_data[..., 0]

        #     #pixel_data = np.clip(pixel_data, 0, pixel_data.max())
        #     #pixel_data = self._undo_global_scale_fn(pixel_data)

        #     if self._proc_config.perform_dicom_scaling:
        #         pixel_data = self._undo_dicom_scale_fn(pixel_data)

        #     self._logger.info("Pixel range after undoing global scale %s %s %s", pixel_data.min(),\
        #     pixel_data.max(), pixel_data.mean())

            # write post processing here
            

    def _undo_reshape(self, pixel_data: np.ndarray) -> np.ndarray:
        """
        Undo zero padding and isotropic resampling based on the flags set in preprocessing

        :param pixel_data: Input pixel data numpy array
        :return: Numpy array after undoing zero padding and isotropic resampling
        """

        undo_methods = self._undo_model_compat_reshape[::-1]
        data = pixel_data[..., 0]

        for method_dict in undo_methods:
            self._logger.info('Applying method %s on pixel_data', method_dict['fn'])

            if method_dict['fn'] == 'undo_zero_pad':
                data = SubtleGADJobType._center_crop(data, method_dict['arg'])
            elif method_dict['fn'] == 'undo_resample':
                data = zoom_interp(data, method_dict['arg'])

        self._logger.info('Final pixel_data shape %s', data.shape)
        pixel_data = data[..., None]
        return pixel_data

    @staticmethod
    #@processify
    def _process_mpr(params: Dict, model) -> np.ndarray:
        """
        Processes single instance of an MPR inference, with the given params dict
        :param params: The input params for the mpr inference with the following keys
         - 'model_dir': Directory path of the model to be loaded
         - 'decrypt_key_hex': The hexadecimal key to decrypt license
         - 'exec_config': The execution config object to be updated in the model
         - 'slices_per_input': Number of context slices for 2.5D processing
         - 'reshape_for_mpr_rotate': Boolean which specifies whether reshape needs to be enabled
            when rotating the input volume
         - 'data': Input data volume
         - 'angle': Angle by which the input data needs to be rotated for MPR processing
         - 'slice_axis': Slice axis of orientation for the input volume
         :return: Prediction from the specified model with the given MPR parameters
        """
        # pylint: disable=global-variable-undefined
        #global gpu_pool
        #gpu_id = gpu_pool.get(block=True)

        try:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"#gpu_id

            #set_keras_memory(allow_growth=True)

            reshape = params['reshape_for_mpr_rotate']

            s1 = time.time()
            input_data = np.copy(params['data'])
            zero_padded = False

            model_input_shape = (
                1, input_data.shape[2], input_data.shape[3], 2 * params['slices_per_input']
            )
            

            if params['angle'] > 0.0 and params['num_rotations'] > 1:
                data_rot = rotate(input_data, params['angle'], reshape=reshape, axes=(1, 2))
            else:
                data_rot = np.copy(input_data)

            #j,k,l,m = data_rot.shape
            #data_rot = np.concatenate([data_rot[1,:,:,:]]*12, axis=0)
            
            #d1 = np.reshape(d1, ())
            #print('d1', d1.shape)
            #data_rot = data_rot.reshape((1,k*12,l,m))

            #data_rot=np.concatenate((data_rot,data_rot), axis=0)
            print('data_rot', data_rot.shape)


            #data_rot =  np.concatenate((data_rot, data_rot, data_rot),axis = 0)
            #print(data_rot.shape,params['slices_per_input'],model._model_config['batch_size'],  params['slice_axis'], input_data.shape[2], input_data.shape[3])
            #print('data_rot',data_rot.shape)
            inf_loader = InferenceLoader(
                input_data=data_rot, slices_per_input=params['slices_per_input'],
                batch_size=1, slice_axis=params['slice_axis'],
                data_order='stack',
                resize=( input_data.shape[2], input_data.shape[3]))
                
            #model._model_config['batch_size']
            #pdb.set_trace()
            #in_load = DataLoader(inf_loader,batch_size = 16,shuffle=False,num_workers=4)
            num_batches = inf_loader.__len__()
            print('num_batches',num_batches)
            Y_pred = []

            for idx in (np.arange(0, num_batches)):
                s1 = time.time()
                X = inf_loader.__getitem__(idx)
                #print('dataloading', time.time() - s1)
                #X = torch.from_numpy(X.astype(np.float32)).to('cuda')
                #print(model._model_obj)
                #print('required input shape', X.shape)
                #print('new input shape', X.shape)
                s2 = time.time()
                Y = model._predict_from_torch_model(X)#.detach().cpu().numpy()#net(X).detach().cpu().numpy()
                #print('data predicting ', time.time() - s2)
                #print('required output shape', Y.shape)
                Y_pred.append(Y[:])
            
            # for i_batch, sample_batched in enumerate(in_load):
            #     print(i_batch, sample_batched.size())
            #     sample_batched = sample_batched.to('cuda')
            #     Y = model._model_obj(sample_batched)
            #     Y = Y.data.float().cpu().numpy()
            #     #if len(Y.shape) > 3:
            #     Y = Y.reshape(-1, input_data.shape[2], input_data.shape[3])
            #     #print('required output shape', Y.shape)
            #     Y_pred.append(Y)



            Y_pred = np.array(Y_pred).reshape((1,num_batches,input_data.shape[2], input_data.shape[3]))
            #print('Y_pred shape ', Y_pred.shape)
            #print('num slices', inf_loader.num_slices)
            Y_pred = Y_pred[:inf_loader.num_slices, ...] # get rid of slice excess

            if params['slice_axis'] == 0:
                #pass
            #elif params['slice_axis'] == 2:
                Y_pred = np.transpose(Y_pred, (1, 0, 2, 3))
            elif params['slice_axis'] == 2:
                Y_pred = np.transpose(Y_pred, (1, 2, 0, 3))
            elif params['slice_axis'] == 3:
                Y_pred = np.transpose(Y_pred, (1, 2, 3, 0))

            if params['num_rotations'] > 1 and params['angle'] > 0:
                Y_pred = rotate(
                    Y_pred, -params['angle'], reshape=False, axes=(1, 2)
                )

            Y_pred = sp.util.resize(Y_pred, (1, input_data.shape[1], input_data.shape[2], input_data.shape[3]))

            print('Model inference', time.time()- s1)
            # np.save(
            #     '/home/srivathsa/projects/studies/gad/all/inference/test/pred_{}_{}.npy'.format(
            #         int(angle), slice_axis
            #     ), Y_pred
            # )
            return Y_pred
        except Exception as e:
            raise Exception('Error in process_mpr', e)
        #finally:
        #    gpu_pool.put(gpu_id)

    def _save_data(self, dict_pixel_data: Dict, dict_template_ds: Dict, out_dicom_dir: str):
        """
        Save pixel data to the output directory
        based on a reference dicom dataset list
        :param dict_pixel_data: dict of array of data by frame to save to output directory
        :param dict_template_ds: dict of template dataset to use to save pixel data
        :param out_dicom_dir: path to output directory
        """
        # save data frame by frame but with the same series UID and a shared UID pool
        # create output directory

        # generate a new series UID

        #np.save(f'{out_dicom_dir}/expected_dicom.npy',self.raw_input_images)
        #np.save(f'{out_dicom_dir}/true_dicom.npy',self.apply_proc_images)

        series_uid = pydicom_utils.generate_uid()
        uid_pool = set()
        uid_pool.add(series_uid)

        out_dicom_dir = os.path.join(out_dicom_dir, series_uid)
        os.makedirs(out_dicom_dir)

        i_instance = 1
        #pdb.set_trace()
        for iframe, frame_seq_name in enumerate(dict_template_ds.keys()):
            # remove unused dimension(s)
            pixel_data = np.squeeze(dict_pixel_data[frame_seq_name])

            # edit pixel data if not_for_clinical_use
            if self._proc_config.not_for_clinical_use:
                pixel_data = pydicom_utils.imprint_volume(pixel_data)

            # get slice position information
            nslices, nrow, ncol = pixel_data.shape
            
            if self._inputs['zd'].isenhanced:
                enhanced_ds_out = self._input_series[0].enhanced_dataset
                nslices = 1
            else:
                enhanced_ds_out = None
            
            # save all individual slices
            for i_slice in range(nslices):
                out_dataset = dict_template_ds[frame_seq_name][i_slice]

                if self._input_series[0].isenhanced:
                    out_dataset = enhanced_ds_out
                # save new pixel data to dataset
                else:
                    slice_pixel_data = pixel_data[i_slice]
                    slice_pixel_data = np.copy(slice_pixel_data).astype(out_dataset.pixel_array.dtype)
                    slice_pixel_data[slice_pixel_data < 0] = 0

                # add model and app info to output dicom file
                for tag in self.private_tag:
                    out_dataset.add_new(*tag)

                template_desc = out_dataset.get("SeriesDescription", "")
                # remove reg matches from the series description
                for reg_match in [
                    series.reg_match for series in self.task.job_definition.series_required
                ]:
                    template_desc = template_desc.replace('_{}'.format(reg_match), '')
                    template_desc = template_desc.replace(reg_match, '')

                # set series-wide metadata
                out_dataset.SeriesDescription = "{}{}{}".format(
                    self._proc_config.series_desc_prefix, template_desc,
                    self._proc_config.series_desc_suffix,
                )
                out_dataset.SeriesInstanceUID = series_uid
                out_dataset.SeriesNumber += self._proc_config.series_number_offset
                out_dataset.Rows = nrow
                out_dataset.Columns = ncol

                # Set Instance UID
                sop_uid = pydicom_utils.generate_uid()
                while sop_uid in uid_pool:
                    sop_uid = pydicom_utils.generate_uid()
                uid_pool.add(sop_uid)

                out_dataset.SOPInstanceUID = sop_uid

                if self._input_series[0].isenhanced and enhanced_ds_out:
                    # save 3D pixel data to enhanced DCM
                    pydicom_utils.save_float_to_enhanced_dataset(
                            enhanced_ds_out,
                            pixel_data,
                            new_rescale=False,
                    )

                    out_path = os.path.join(
                            out_dicom_dir,
                            "IMG-{:04d}-{:04d}.dcm".format(iframe, i_slice)
                    )
                    
                    enhanced_ds_out.save_as(out_path)

                else:

                    out_dataset.PixelData = slice_pixel_data.tostring()

                    # save in output folder
                    out_path = os.path.join(
                        out_dicom_dir,
                        "IMG-{:04d}-{:04d}.dcm".format(iframe, i_slice),
                    )
                    out_dataset.save_as(out_path)
                    i_instance += 1
