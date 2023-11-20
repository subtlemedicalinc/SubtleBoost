"""The SubtleBoost jobs

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
import tempfile
import multiprocessing
from multiprocessing import Pool, Queue
from functools import partial

from subtle.util.inference_job_utils import (
    BaseJobType, GenericInferenceModel
)
from subtle.util.data_loader import InferenceLoader
from subtle.util.multiprocess_utils import processify
from subtle.procutil import preprocess_single_series
from subtle.procutil import registration_utils
from subtle.dcmutil import pydicom_utils, series_utils

from subtle.procutil.registration_utils import register_im

import SimpleITK as sitk
import pydicom

from subtle.procutil.segmentation_utils import HDBetInMemory

import torch
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
import tqdm
import pdb
import torch
import HD_BET
import warnings
import time
warnings.filterwarnings("ignore")
torch.manual_seed(0)

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
class SubtleBoostJobType(BaseJobType):
    """The job type of SubtleBoost dose reduction"""
    default_processing_config_boost = {
        # app params
        "app_name": "SubtleBOOST",
        "app_id": 12000,
        "app_version": "0.0.0",
        "min_gpu_mem_mb": 9800.0,
        "require_GPU": True,

        # model params
        "model_id": "None",

        # general params
        "not_for_clinical_use": False,
        "duplicate_series": False,
        "make_derived":True,

        # preprocessing params:
        #pipeline definition - custom config 
        "model_type": "boost_process",
        "pipeline_preproc": {
            'boost_process' : {
                'STEP1' : {'op': 'MASK'},
                'STEP2' : {'op': 'SKULLSTRIP'},
                'STEP3' : {'op' : 'REGISTER'},
                'STEP4' : {'op': 'SCALEGLOBAL'}
            }
            
        },
        "pipeline_postproc": {
            'boost_process' : {
                'STEP1' : {'op' : 'CLIP'},
                'STEP2' : {'op' : 'RESCALEGLOBAL'}

            }
        },

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
        "series_desc_prefix": "",
        "series_desc_suffix": "",
        "series_number_offset": 100,
        "series_description":"",
        "uid_prefix":"",
        "study_description_suffix" : "",
        "protocol_name_suffix" : "",
    }

    #def SubtleBoostProcConfig(self):
    SubtleBoostProcConfig = namedtuple(
            "SubtleBoostProcConfig", ' '.join(list(default_processing_config_boost.keys()))
        )

        #return SubtleBoostProcConfig

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
                         "noise_mask_threshold": 0.08,
                         "noise_mask_selem": False}, {}),
            'SKULLSTRIP': (self.apply_brain_mask, {}, {}),
            'REGISTER': (self._register, 
                        {"transform_type" : "affine",
                         "use_mask_reg" : True,
                         "reg_n_levels": 4}, {}),
            'HIST': (self._match_histogram,
                        {}, {}),
            'SCALETAG': (self._scale_intensity_with_dicom_tags,
                        {}, {}),
            'SCALEGLOBAL': (self._scale_intensity,
                        {"num_scale_context_slices" : 20,
                         "joint_normalize": False, 
                         "scale_ref_zero_img": False}, {}),
            'CLIP': (self.clip_boost, {'lb' : 0}, {}),
            'RESCALEDICOM':(self._undo_dicom_scale_fn, 
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
        proc_config = self.default_processing_config_boost.copy()
        proc_config.update(task.job_definition.exec_config)

        proc_config = self._parse_pipeline_definition(proc_config)

        k = list(proc_config.keys())
        _ = [proc_config.pop(f) for f in k if f not in self.SubtleBoostProcConfig._fields]
        self._proc_config = self.SubtleBoostProcConfig(**proc_config)
        

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
        for op_name in ['RESCALEGLOBAL']:
            op_in_preproc, step_num_pre, step_pre = self._is_step_in_pipeline(op_name, proc_config["pipeline_postproc"])
            op_in_postproc, step_num_post, step_post = self._is_step_in_pipeline('SCALEGLOBAL', proc_config["pipeline_preproc"])

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
    
    def clip_boost(self,input_data: np.ndarray, param: dict, raw_input_images: np.ndarray = None):
        if raw_input_images is not None:

            return np.clip(input_data,param['lb'], input_data.max()), np.clip(raw_input_images,param['lb'], input_data.max())
        else:
            return np.clip(input_data, param['lb'], input_data.max())

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
        for frame_seq_name, data_array in self._raw_input.items():
            # check the shape, if it is not square, pad to square
            # because the saved model need a fixed receptive field
            input_data= data_array.copy()  # better to be deep.copy()

            for i in range(1, len(self._proc_config.pipeline_preproc)+1):
                # get the step info
                step_dict = self._proc_config.pipeline_preproc['STEP{}'.format(i)]
                # get the function to run for this step
                func = self.function_keys[step_dict['op']][0]
                # run function
                raw_input_images = np.copy(input_data)
                if step_dict['op'] == "MASK":
                    input_data, mask = func(input_data, step_dict['param'])
                    raw_input_images = np.copy(input_data)
                    self.mask = np.copy(mask)
                else:
                    input_data, raw_input_images = func(input_data, step_dict['param'], raw_input_images)

            # assign value per frame
            dict_input_data[frame_seq_name] = raw_input_images

        return dict_input_data
    
    def _check_acquisition_type(self):
        if self._inputs['zd'].acquisition_type == '2D':
            self._proc_config = self._proc_config._replace(inference_mpr = False)
            self._proc_config = self._proc_config._replace(reshape_for_mpr_rotate = False)
            self._proc_config = self._proc_config._replace(num_rotations = 1)
            #self._proc_config = self._proc_config._replace(model_resolution = None)
            self._proc_config = self._proc_config._replace(pipeline_preproc = {'STEP1' : {'op': 'MASK', 'param' : {"noise_mask_area" : False, "noise_mask_threshold": 0.1, "noise_mask_selem" : False}}, 
                                                  'STEP2' : {'op' : 'REGISTER', 'param' : {"transform_type": "affine", "use_mask_reg": False, "reg_n_levels": 4}}, 
                                                  'STEP3':  {'op' : 'HIST', 'param' : {}},
                                                  'STEP4' : {'op' : 'SCALEGLOBAL', 'param' : {"joint_normalize" : True, "num_scale_context_slices" : 3, "scale_ref_zero_img": False}}})
    
    def _preprocess(self) -> Dict:
        """
        Preprocess function called by the __call__ method of BaseJobType. Before calling the actual
        preprocessing logic, this method calls a bunch of initialization functions to get DICOM
        data, set manufacturer specific config, get the DICOM pixel spacing, get the DICOM scaling
        information and finally get the raw pixel data

        """
        self._get_input_series()

        self._check_acquisition_type()

        self._get_pixel_spacing()

        self._get_dicom_scale_coeff()

        dict_pixel_data = self._preprocess_pixel_data()

        return dict_pixel_data

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-locals
    #@processify
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
        del self._itk_data
        del self._raw_input
        del self.mask
        
        dict_output_array = {}

        self._logger.info("starting inference (job type: %s)", self.name)

        model_input_shape = (
                1, dict_pixel_data['single_frame'].shape[2], dict_pixel_data['single_frame'].shape[3], 2 * self._proc_config.slices_per_input
            )

        self.model = GenericInferenceModel(
                model_dir=self.model_dir, decrypt_key_hex=self.decrypt_key_hex,
                input_shape=model_input_shape
            )
            
        self.model.update_config(self.task.job_definition.exec_config)

        (dict_pixel_data['single_frame'], self._undo_model_compat_reshape) = self._process_model_input_compatibility(dict_pixel_data['single_frame'])
        
        dict_pixel_data['single_frame'] = np.clip(dict_pixel_data['single_frame'], 0, dict_pixel_data['single_frame'].max())

        for frame_seq_name, pixel_data in dict_pixel_data.items():
            # perform inference with default input format (NHWC)

            angle_start = self._proc_config.mpr_angle_start
            angle_end = self._proc_config.mpr_angle_end
            num_rotations = self._proc_config.num_rotations

            mpr_params = []

            _, n_slices, height, width = self._resampling_size
            flag_largecase = False
            if n_slices> 400:
                flag_largecase = True

            param_obj = {
                'model_dir': self.model_dir,
                'decrypt_key_hex': self.decrypt_key_hex,
                'exec_config': self.task.job_definition.exec_config,
                'inference_mpr': self._proc_config.inference_mpr,
                'slices_per_input': self._proc_config.slices_per_input,
                'reshape_for_mpr_rotate': self._proc_config.reshape_for_mpr_rotate,
                'num_rotations': self.task.job_definition.exec_config['num_rotations'],
                'batch_size' : 8
            }

            if flag_largecase:
                param_obj['num_workers'] = 1
                param_obj['batch_size'] = 1
            else:
                param_obj['num_workers'] = 4    
                param_obj['batch_size'] = 1

            if param_obj['inference_mpr']:
                slice_axes = [0, 2, 3]
            else:
                slice_axes = [0]

            if param_obj['inference_mpr'] and param_obj['num_rotations'] > 1:
                angles = np.linspace(0, 90, param_obj['num_rotations'], endpoint=False)
            else:
                angles = [0]
            predictions = []
            data= {}
            
            from collections import deque
            def rotate_func(pixel_data, new_angle, q):
                if q:
                    q.put(rotate(pixel_data, new_angle, (1,2),self._proc_config.reshape_for_mpr_rotate))
                return rotate(pixel_data, new_angle, (1,2),self._proc_config.reshape_for_mpr_rotate)


            next_data = np.copy(pixel_data)
            curr_angle = 0
            queue = []
            if len(angles) > 1:
                queue = deque(angles[1:])
            p1 = None
            mpq = multiprocessing.Queue()
            slice_results = None
            Y_run_sum = None
            multiprocess_q = [None]
            mpqs = [None]
            while next_data is not None:
                if len(queue) > 0 and not flag_largecase:
                    new_angle = queue.popleft()
                    p1 = multiprocessing.Process(target = rotate_func, args=(pixel_data, new_angle, mpq))
                    p1.start()

                angle = curr_angle
                param_obj['size'] = [0] + list(next_data.shape[1:])
                param_obj['all_axes'] = slice_axes
                mpr_params = []
                Y_pred = []
                for slice_axis in slice_axes:
                    pobj = copy.deepcopy(param_obj)
                    pobj['angle'] = angle
                    pobj['slice_axis'] = slice_axis
                    pobj['reshape'] = [pixel_data.shape[1], pixel_data.shape[2], pixel_data.shape[3]]
                    mpr_params = copy.deepcopy(pobj)
                    pobj['data'] = copy.deepcopy(next_data)
                    Y_pred = np.array(SubtleBoostJobType._process_mpr(pobj, self.model))
                    
                    if flag_largecase:
                        slice_results, Y_run_sum=  SubtleBoostJobType._reorient_output(Y_pred,slice_results, Y_run_sum, mpr_params)
                    else:
                        mpqs.append(multiprocessing.Queue())
                        multiprocess_q.append(multiprocessing.Process(target = SubtleBoostJobType.pool_model, args=(Y_pred,mpr_params,mpqs[-2],multiprocess_q[-1], mpqs[-1])))
                        multiprocess_q[-1].start()
                if queue and flag_largecase:
                    new_angle = queue.popleft()
                    next_data = rotate_func(pixel_data, new_angle, None)
                    curr_angle = new_angle
                elif p1 is not None and not flag_largecase:
                    next_data = mpq.get()
                    p1.join()
                    p1.close()
                    curr_angle = new_angle 
                    p1 = None
                else:
                    next_data = None 
                    curr_angle = None

            if not flag_largecase:
                slice_results , Y_run_sum = mpqs[-1].get()
                multiprocess_q[-1].join()

            if num_rotations > 1:
                slice_results = np.divide(
                        np.sum(slice_results, axis=(0, 1), keepdims=False),
                        Y_run_sum,
                        where=Y_run_sum > 0
                    )
            else:
                slice_results = slice_results.squeeze()

            del self.model 
            del next_data
            del Y_pred
            del Y_run_sum 

            slice_results = sp.util.resize(slice_results[..., None], (n_slices,height,width,1))
            # undo zero padding and isotropic resampling
            slice_results = self._undo_reshape(slice_results)
            dict_output_array[frame_seq_name] = slice_results

        self._logger.info("inference finished")

        
        return dict_output_array

    # pylint: disable=arguments-differ
    def _postprocess(self, dict_pixel_data: Dict, out_dicom_dir: str):
        """
        Undo some of the preprocessing (scaling) steps for the DICOM pixel data to match that
        of the input data. This method is called by the __call__ method of the BaseJobType

        :param dict_pixel_data: dict of the result pixel data in a numpy array by frame.
        :param out_dicom_dir: the directory or tarball of output DICOM files
        """

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

        self._save_data(self._output_data, self._input_datasets[1], out_dicom_dir)

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

        zero_dose_series = [(series) for series in self._input_series  if 'BPRE' in series.seriesdescription.upper()][0]
        low_dose_series =  [(series) for series in self._input_series  if 'BPOST' in series.seriesdescription.upper()][0]

        self._inputs = {'zd' : zero_dose_series, 'ld': low_dose_series}

        #Check for one zero dose, low dose
        zero_dose_pixels = [list(zero_dose_series.get_pixel_data(rescale=False).values())][0][0]
        low_dose_pixels = [list(low_dose_series.get_pixel_data(rescale=False).values())][0][0]    


        self._input_datasets = (zero_dose_series.get_dict_sorted_datasets(),
                                low_dose_series.get_dict_sorted_datasets()
        )

        #for frame_seq_name in self._input_datasets[0].keys():
        self._raw_input['single_frame'] = np.array([zero_dose_pixels, low_dose_pixels])
        self._logger.info("the shape of array %s", self._raw_input['single_frame'].shape)


        try:
            self._itk_data['zero_dose'] = zero_dose_series._list_itks[0]
            self._itk_data['low_dose'] = low_dose_series._list_itks[0]
        except:
            
            self._itk_data['zero_dose'] = sitk.GetImageFromArray(zero_dose_pixels)
            self._itk_data['low_dose'] = sitk.GetImageFromArray(low_dose_pixels)

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

        return images, np.array(mask)

    #@processify
    def apply_brain_mask(self, ims: np.ndarray, param: dict, raw_input_images:np.ndarray=None):
        self._logger.info("Performing skull stripping...")

        brain_mask = self._brain_mask(ims)

        if True:
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
    def _rescale_slope_intercept(input_data,rescale_slope: float, rescale_intercept: float, scale_slope: float
    ) -> np.ndarray:
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
        
        return (input_data * rescale_slope * scale_slope - rescale_intercept) / rescale_slope

    def _scale_intensity_with_dicom_tags(self, scaled_images: np.ndarray, param:dict, raw_scaled_images: np.ndarray=None) -> np.ndarray:
        """
        Helper function to scale the image intensity using DICOM header information

        :param images: Zero and low dose images as numpy array
        :return: The scaled images as numpy array
        """
        #if self._proc_config.perform_dicom_scaling and self._has_dicom_scaling_info():
        self._logger.info("Performing intensity scaling using DICOM tags...")

        scale_fn = lambda idx: (
            lambda ims: SubtleBoostJobType._scale_slope_intercept(ims[idx],
            **self._dicom_scale_coeff[idx])
        )

        scaled_images[0] = scale_fn(0)(scaled_images)
        scaled_images[1] = scale_fn(1)(scaled_images)

        raw_scaled_images[0] = scale_fn(0)(raw_scaled_images)
        raw_scaled_images[1] = scale_fn(1)(raw_scaled_images)

        return scaled_images, raw_scaled_images

    def _undo_dicom_scale_fn(self, img, param):
        return SubtleBoostJobType._rescale_slope_intercept(img,**self._dicom_scale_coeff[0])


    def _get_pixel_spacing(self):
        """
        Obtains the pixel spacing information from DICOM header of input dataset and sets it to the
        local pixel spacing attribute
        """

        x_zero, y_zero = self._inputs['zd'].pixelspacing
        z_zero = self._inputs['zd'].slicethickness

        x_low, y_low = self._inputs['ld'].pixelspacing
        z_low = self._inputs['ld'].slicethickness

        self._pixel_spacing = [(x_zero, y_zero, z_zero), (x_low, y_low, z_low)]
    
    def apply_reg_transform(self,img, spacing, transform_params, ref_img=None):
        simg = sitk.GetImageFromArray(img)
        simg.SetSpacing(spacing)

        if ref_img:
            simg.CopyInformation(ref_img)

        params = transform_params[0]

        simg_trans = sitk.Transformix(simg, params)
        simg_arr = sitk.GetArrayFromImage(simg_trans)
        return simg_arr

    def _register(self, reg_images: np.ndarray,param: dict, raw_reg_images: np.ndarray=None) -> np.ndarray:
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

        self._logger.info("Performing registration of low dose with respect to zero dose...")
        spars = sitk.GetDefaultParameterMap(param["transform_type"], param['reg_n_levels'])

        # register the low dose image with respect to the zero dose image
        reg_args = {
            'im_fixed_spacing': self._pixel_spacing[0],
            'im_moving_spacing': self._pixel_spacing[1],
            'param_map': spars,
            'max_iter': 400
        }

        reg_images[1], reg_params = registration_utils.register_im(
            im_fixed=reg_images[0], im_moving=reg_images[1], ref_fixed = self._itk_data['zero_dose'], ref_moving=self._itk_data['low_dose'],  **reg_args
        )
        
        idty_fn = lambda ims: ims[0]

        if param["use_mask_reg"]:
            apply_reg = lambda ims: self.apply_reg_transform(ims[1], self._pixel_spacing[1], reg_params, ref_img=self._itk_data['zero_dose'] )
        else:
            self._logger.info('Registration will be re-run for unmasked images...')
            apply_reg = lambda ims: registration_utils.register_im(
                im_fixed=ims[0], im_moving=ims[1], ref_fixed = self._itk_data['zero_dose'], ref_moving=self._itk_data['low_dose'], return_params=False, **reg_args
            )
        
        raw_reg_images[1] = apply_reg(raw_reg_images)

        return reg_images, raw_reg_images
    

    def _match_histogram(self, hist_images: np.ndarray, param: dict, raw_hist_images: np.ndarray =None) -> np.ndarray:
        """
        Performs histogram matching of low dose by having zero dose as reference
        :param images: Input zero dose and low dose images as numpy array
        :return: Histogram matched images as numpy array
        """
        self._logger.info("Matching histogram of low dose with respect to zero dose...")

        hist_images[1] = preprocess_single_series.match_histogram(
            img=hist_images[1], ref_img=hist_images[0], levels = 1024, points=50, mean_intensity=True
        )

        idty_fn = lambda ims: ims[0]
        hist_match_fn = lambda ims: preprocess_single_series.match_histogram(ims[1], ims[0], levels = 1024, points=50, mean_intensity=True )

        raw_hist_images[1] = hist_match_fn(raw_hist_images)

        return hist_images, raw_hist_images

    def _scale_intensity(self, scale_images: np.ndarray, param: dict, raw_scale_images: np.ndarray= None) -> np.ndarray:
        """
        Performs relative and global scaling of images by using the pixel values inside the
        noise mask previously computed

        :param images: Input zero dose and low dose images as numpy array
        :param noise_mask: Binary noise mask computed in an earlier step of pre-processing
        :return: Scaled images as numpy array
        """

        self._logger.info("Performing intensity scaling...")
        num_slices = scale_images.shape[1]

        idx_center = range(
            num_slices//2 - param["num_scale_context_slices"]//2,
            num_slices//2 + param["num_scale_context_slices"]//2
        )

        idx_center = np.clip(idx_center, 0, num_slices-1)

        # use pixels inside the noise mask of zero dose
        ref_mask = self.mask[0, idx_center,:,:]
        context_img_zero = scale_images[0, idx_center, :,:][ref_mask != 0].ravel()
        context_img_low = scale_images[1, idx_center, :,:][ref_mask != 0].ravel()
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
            global_scale = np.round(np.expand_dims(global_scale, axis=a), 2)
        

        # repeat the computed scale such that its shape is always (1, 2)
        if global_scale.ndim == 1:
            global_scale = np.repeat([global_scale], repeats=scale_images.shape[1], axis=1)
        elif global_scale.ndim == 2 and global_scale.shape[1] == 1:
            global_scale = np.repeat(global_scale, repeats=scale_images.shape[1], axis=1)
        
        self._logger.info("Computed global scale %s with shape %s", global_scale, \
        global_scale.shape)

        scale_images /= global_scale[:,:,None,None]

        global_scale = global_scale.transpose(1,0)

        self.global_scale = global_scale[:, :, None, None]
        global_scale_fn = lambda ims: (ims / self.global_scale)

        raw_scale_images = global_scale_fn(raw_scale_images)
        self.global_scale = self.global_scale[0,:][0][0][0]

        return scale_images.transpose(1, 0, 2, 3), raw_scale_images

    def _undo_global_scale_fn(self,img, param):
        return img*self.global_scale

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
        model_input = self.model._model_config['input_shape']#model._model_obj.inputs[0]
        model_shape = (int(model_input[1]), int(model_input[2]))
        input_shape = (input_data.shape[-2], input_data.shape[-1])

        pixel_spacing = np.round(np.array(self._pixel_spacing[0])[::-1], decimals=2)

        if self._proc_config.model_resolution and not np.array_equal(pixel_spacing, self._proc_config.model_resolution):
            self._logger.info('Resampling to isotropic resolution...')

            resize_factor = (pixel_spacing / self._proc_config.model_resolution)
            new_shape = np.round(input_data[0].shape * resize_factor)
            real_resize_factor = new_shape / input_data[0].shape
            from contextlib import closing
            with closing(multiprocessing.Pool(maxtasksperchild=1)) as pool:
                results = pool.map(partial(zoom_interp, zoom =real_resize_factor), [input_data[0], input_data[1]])
            input_data = np.array(results)
            undo_factor = 1.0 / real_resize_factor
            undo_methods.append({
                'fn': 'undo_resample',
                'arg': undo_factor
            })
        
            self._logger.info('Input data shape after resampling %s', input_data.shape)
        self._resampling_size = input_data.shape

        if (input_data.shape[-2], input_data.shape[-1]) != model_shape:
            self._logger.info('Zero padding to fit model shape...')
            curr_shape = input_data.shape
            input_data = SubtleBoostJobType._zero_pad(
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
        if (input_data.shape[-2], input_data.shape[-1]) != model_shape:
            input_data = sp.util.resize(input_data, (input_data.shape[0], input_data.shape[1],model_shape[0], model_shape[1]))

            self._logger.info('Input data shape after resizing %s', input_data.shape)

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
                    start, end = SubtleBoostJobType._get_crop_range(diff)
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
                if (sh - img.shape[i]) % 2 == 0:
                    diff = (sh - img.shape[i]) // 2 
                    npad.append((diff, diff))
                else:
                    diff = ((sh - img.shape[i]) // 2 )
                    npad.append((diff+1, diff))

        return np.pad(img, pad_width=npad, mode='constant', constant_values=const_val)
    
    #@staticmethod
    #@processify
    def _mask_npy(img_npy, dcm_ref, hdbet):
        mask = None
        
        data_sitk = sitk.GetImageFromArray(img_npy)
        
        data_sitk.CopyInformation(dcm_ref)
        
        mask = hdbet.run_hd_bet_inmem(data_sitk)

        return mask

    def _brain_mask(self,ims_proc):
        mask = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ## Temporarily using DL based method for extraction
        torch.manual_seed(0)
        hdbet = HDBetInMemory(mode = 'fast', model_path = self.model_dir, do_tta = False)

        mask_zero = (SubtleBoostJobType._mask_npy(ims_proc[0,:, ...], self._itk_data['zero_dose'], hdbet))

        mask_low = (SubtleBoostJobType._mask_npy(ims_proc[1,:,  ...], self._itk_data['low_dose'], hdbet))
        
        mask = np.array([mask_zero, mask_low]).transpose(0,1, 2, 3)

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
                self._logger.info('Applying method %s on pixel_data', method_dict['fn'] )
                data = SubtleBoostJobType._center_crop(data, method_dict['arg'])
            elif method_dict['fn'] == 'undo_resample':
                data = zoom_interp(data, method_dict['arg'])

        self._logger.info('Final pixel_data shape %s', data.shape)
        pixel_data = data[..., None]
        return pixel_data
    
    @staticmethod
    def _reorient_output(Y_pred, slice_results, Y_run_sum, params):
        if params['slice_axis'] == 0:
            pass
        elif params['slice_axis'] == 2:
            Y_pred = np.transpose(Y_pred, (1,0,2))#(1, 2, 0, 3))
        elif params['slice_axis'] == 3:
            Y_pred = np.transpose(Y_pred, (1,2,0))#(1, 2, 3, 0))

        if params['num_rotations'] > 1 and params['angle'] > 0:
            Y_pred = rotate(
                Y_pred, -params['angle'], reshape=False, axes=(0, 1)
            )
        Y_pred = sp.util.resize(Y_pred, (1,1, params['reshape'][0], params['reshape'][1], params['reshape'][2]))

        Y_pred = np.clip(Y_pred, 0, Y_pred.max())
        Y_masks_sum = np.sum(np.array(Y_pred > 0, dtype=np.int), axis=(0, 1), keepdims=False)
        
        if Y_run_sum is not None:
            Y_masks_sum = np.add(Y_masks_sum, Y_run_sum)
        if slice_results is not None:
            Y_pred = np.add(Y_pred , slice_results)
        
        return Y_pred, Y_masks_sum

    @staticmethod
    def pool_model(Y_pred,params,prevmpq,prevmultipq, mpq):
        slice_results, Y_run_sum = None, None
        if prevmpq is not None:
            val = prevmpq.get()
            slice_results, Y_run_sum = val[0] , val[1]
            prevmultipq.terminate()

        slice_results, Y_run_sum = SubtleBoostJobType._reorient_output(Y_pred, slice_results, Y_run_sum, params)

        mpq.put([slice_results, Y_run_sum])

    @staticmethod
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
        try:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"#gpu_id

            reshape = params['reshape_for_mpr_rotate']

            zero_padded = False

            model_input_shape = (
                1, params['data'].shape[2], params['data'].shape[3], 2 * params['slices_per_input']
            )

            inf_loader = InferenceLoader(
                input_data=params['data'], slices_per_input=params['slices_per_input'],
                batch_size=params['batch_size'], slice_axis=params['slice_axis'],
                data_order='stack',
                resize=( params['reshape'][1], params['reshape'][2]))
                
            in_load = DataLoader(inf_loader,batch_size = 8,shuffle=False,num_workers=params['num_workers'])
            num_batches = inf_loader.__len__()
            del params['data']

            Y_pred = []
            
            for i_batch, sample_batched in enumerate(in_load):
                sample_batched = sample_batched.to('cuda')
                Y = model._model_obj(sample_batched).detach().cpu().numpy() 
                Y = np.squeeze(Y)
                Y = np.reshape(Y, (-1, params['reshape'][1], params['reshape'][2]))
                Y_pred.extend(Y[:,:,:])
            
            Y_pred = np.array(Y_pred)
            Y_pred = Y_pred[:inf_loader.num_slices, ...] # get rid of slice excess
            
            return Y_pred
        except Exception as e:
            raise Exception('Error in process_mpr', e)

    def _update_common_metadata(self, ds: pydicom.Dataset, series_uid: str,
                                nrow: int = None, ncol: int = None, uid_pool: set = None):
        """
        Set series wide metadata to a dataset. Can be either a regular DICOM dataset
        (needs to be called for each slice) or a single enhanced DICOM dataset.
        :param ds: pydicom dataset to update
        :param series_uid: new SeriesInstanceUID
        :param nrow: Number of rows
        :param ncol: umber of columns
        :return:
        """
        # add model and app info to output dicom file
        for tag in self.private_tag:
            ds.add_new(*tag)

        # set series-wide metadata, SeriesDescription Update
        if not self._proc_config.series_description:
            ld_series_description = self._inputs['ld'].seriesdescription.upper().replace('BPOST', "")
            ds.SeriesDescription = "{}{}{}".format(
                    self._proc_config.series_desc_prefix, ld_series_description,
                    self._proc_config.series_desc_suffix,
            )
        else:
            ds.SeriesDescription = self._proc_config.series_description

        #SOPInstanceUID update
        sop_uid_in = ds.SOPInstanceUID if ds.SOPInstanceUID else str(i+1)
        sop_uid = pydicom_utils.generate_uid(random_uid=self._proc_config.duplicate_series,
                                                        input_uid=sop_uid_in,
                                                        manufacturer=self._input_series[0].manufacturer,
                                                        custom_prefix=self._proc_config.uid_prefix)

        while sop_uid in uid_pool:
            sop_uid = pydicom_utils.generate_uid(random_uid=self._proc_config.duplicate_series,
                                                            input_uid=sop_uid,
                                                            manufacturer=self._input_series[0].manufacturer,
                                                            custom_prefix=self._proc_config.uid_prefix)
        uid_pool.add(sop_uid)

        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber += self._proc_config.series_number_offset
        ds.Rows = nrow if nrow else ds.Rows
        ds.Columns = ncol if ncol else ds.Columns
        if self._proc_config.make_derived and 'ImageType' in ds and len(ds.ImageType) > 0:
            ds.ImageType[0] = 'DERIVED'

        ds.SOPInstanceUID = sop_uid

    def _save_data(self, dict_pixel_data: Dict, dict_template_ds: Dict, out_dicom_dir: str):
        """
        Save pixel data to the output directory
        based on a reference dicom dataset list
        :param dict_pixel_data: dict of array of data by frame to save to output directory
        :param dict_template_ds: dict of template dataset to use to save pixel data
        :param out_dicom_dir: path to output directory
        """
        # save data frame by frame but with the same series UID and a shared UID pool
        self._logger.info("Saving data...")
        
        uid_pool = set()
        if not self._input_series[0].seriesinstanceUID and not self._proc_config.duplicate_series:
            self._logger.warning("The input SeriesUID was empty, the output SeriesUID will not be unique. "
                                 "This series can be overridden by any other series processed with these conditions. "
                                 "Consider using `duplicate_series: True` in the app config.")
        
        #SeriesInstanceUID update
        series_uid = pydicom_utils.generate_uid(random_uid=self._proc_config.duplicate_series,
                                                input_uid=self._input_series[0].seriesinstanceUID,
                                                custom_prefix=self._proc_config.uid_prefix)

        uid_pool.add(series_uid)

        out_dicom_dir = os.path.join(out_dicom_dir, series_uid)
        os.makedirs(out_dicom_dir)

        i_instance = 1
        for iframe, frame_seq_name in enumerate(dict_template_ds.keys()):
            # remove unused dimension(s)
            pixel_data = np.squeeze(dict_pixel_data[frame_seq_name])

            bits_stored = self._inputs["ld"].bits_stored
            if self._inputs["ld"].pixel_representation != 0:
                t_min, t_max = (-(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1)
            else:
                t_min, t_max = 0, (1 << bits_stored) - 1

            pixel_data[pixel_data < t_min] = t_min
            pixel_data[pixel_data > t_max] = t_max

            # edit pixel data if not_for_clinical_use
            if self._proc_config.not_for_clinical_use:
                pixel_data = pydicom_utils.imprint_volume(pixel_data)

            # get slice position information
            nslices, nrow, ncol = pixel_data.shape
            
            if self._inputs['zd'].isenhanced:
                enhanced_ds_out = self._input_series[1].enhanced_dataset
                nslices = 1
            else:
                enhanced_ds_out = None
            
            # save all individual slices
            for i_slice in range(nslices):
                out_dataset = dict_template_ds[frame_seq_name][i_slice]

                #check for enhanced_dataset
                if self._input_series[1].isenhanced and enhanced_ds_out:
                    out_dataset = enhanced_ds_out
                    self._update_common_metadata(out_dataset,series_uid, nrow, ncol, uid_pool)

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

                # save new pixel data to dataset
                else:
                    slice_pixel_data = pixel_data[i_slice]
                    #slice_pixel_data[slice_pixel_data < 0] = 0
                    slice_pixel_data = np.copy(slice_pixel_data).astype(out_dataset.pixel_array.dtype)
                    self._update_common_metadata(out_dataset,series_uid, nrow, ncol, uid_pool)

                    out_dataset.PixelData = slice_pixel_data.tostring()

                    # if dicom is compressed, decompress it to be able modify pixel data and write output
                    if out_dataset.file_meta.TransferSyntaxUID.is_compressed:
                        out_dataset.decompress()

                    # save in output folder
                    out_path = os.path.join(
                        out_dicom_dir,
                        "IMG-{:04d}-{:04d}.dcm".format(iframe, i_slice),
                    )
                    out_dataset.save_as(out_path)
                    i_instance += 1
