"""
Unit test for inference and postprocess in SubtleGad jobs

@author: Srivathsa Pasumarthi <srivathsa@subtlemedical.com>
Copyright Subtle Medical, Inc. (https://subtlemedical.com)
Created on 2020/03/03
"""

import os
import tempfile
import shutil
from collections import namedtuple
from glob import glob
import unittest
from unittest.mock import MagicMock
import pytest
import numpy as np
import pydicom
from subtle.dcmutil.series_io import dicomscan
from subtle.util.inference_job_utils import GenericInferenceModel
# pylint: disable=import-error
import subtle_gad_jobs
import mock
import logging
import torch

torch.manual_seed(0)
@pytest.mark.processing
class ProcessingTest(unittest.TestCase):
    """
    A unittest framework to test the preprocessing functions
    """

    def setUp(self):
        """
        Setup proc config and init params for job class
        """
        processing_config = {
            "app_name": "SubtleGAD",
            "app_id": 4000,
            "app_version": "unittest",
            "model_id":"20230921105336-unified",
            "not_for_clinical_use": False,
            "inference_mpr": False,
            "num_rotations": 1,
            "skip_mpr": False,
            "slices_per_input": 7,
            "mpr_angle_start": 0,
            "mpr_angle_end": 90,
            "reshape_for_mpr_rotate": True,
            "num_procs_per_gpu": 2,
            "series_desc_prefix": "SubtleGAD:",
            "series_desc_suffix": "",
            "series_number_offset": 100,
            "model_resolution": [0.5, 0.5, 0.5],
            "min_gpu_mem_mb": 9800.0,
            "allocate_available_gpus": True,
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
        }

        self.path_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
        self.model_dir = os.path.join(self.path_data, "model", "20230921105336-unified")

        self.processing_config = processing_config

        self.mock_task = MagicMock()
        self.mock_task.job_name = 'test'
        exec_config_keys = [
            'app_name', 'app_id', 'app_version', 'model_id', 'not_for_clinical_use',
            'series_desc_prefix', 'series_desc_suffix', 'series_number_offset', 'num_rotations', 'inference_mpr', 'model_type', 'pipeline_preproc', 'pipeline_postproc'
        ]
        self.mock_task.job_definition.exec_config = {
            k: v for k, v in processing_config.items() if k in exec_config_keys
        }

        self.job_obj = subtle_gad_jobs.SubtleGADJobType(
            task=self.mock_task, model_dir=self.model_dir
        )

        self.zero_path_dicom = os.path.join(self.path_data, "IM001", "OSag_3D_T1BRAVO_zero_dose_7")
        self.zero_series_in = list(dicomscan(self.zero_path_dicom, methodname="itk").values())[0]

        self.low_path_dicom = os.path.join(self.path_data, "IM001", "OSag_3D_T1BRAVO_low_dose_8")
        self.low_series_in = list(dicomscan(self.low_path_dicom, methodname="itk").values())[0]

        self.list_filename_in_zero = [os.path.join(self.zero_path_dicom, f)
                                 for f in os.listdir(self.zero_path_dicom)
                                 if f.endswith('.dcm')]

        self.list_filename_in_low = [os.path.join(self.low_path_dicom, f)
                                    for f in os.listdir(self.low_path_dicom)
                                    if f.endswith('.dcm')]

        self.nslices_zero = len(self.list_filename_in_zero)
        self.nslices_low = len(self.list_filename_in_low)

        self.sequence_name = self.zero_series_in.get_frame_names()[0]

        self.job_obj.task.dict_required_series = {
            'zero_dose_ge': self.zero_series_in,
            'low_dose_ge': self.low_series_in
        }

        

        #self.dict_pixel_data = self.job_obj._preprocess()
        self.job_obj._get_input_series()
        self.job_obj._get_pixel_spacing()
        self.job_obj._get_dicom_scale_coeff()
        #self.job_obj._get_raw_pixel_data()



    
    @pytest.mark.ver3
    @pytest.mark.req16
    def _validate_dicom_datasets(self):
        '''
        Comparing if zero & low dose input series have the same number of slices and have the same slice locations.
        '''

        #Assume constant frame name
        dict_template_ds_zero = list(self.job_obj._input_series[0].get_dict_sorted_datasets()[self.sequence_name])
        dict_template_ds_low = list(self.job_obj._input_series[1].get_dict_sorted_datasets()[self.sequence_name])

        # check number of slices
        self.assertEqual(
            len(dict_template_ds_zero),
            self.nslices_low,
            "Wrong number of slices for zero dose series",
        )

        self.assertEqual(
            len(dict_template_ds_zero),
            self.nslices_low,
            "Wrong number of slices for low dose series",
        )
        # Check T1 list is sorted
        list_slice_loc_zero = [
            dcm.SliceLocation for dcm in dict_template_ds_zero
        ]

        expected_list_slice_loc = sorted(
            pydicom.read_file(f, stop_before_pixels=True).SliceLocation
            for f in self.list_filename_in_zero
        )
        #
        self.assertEqual(
            list_slice_loc_zero,
            expected_list_slice_loc,
            "zero dose Dataset list is not sorted",
        )

        list_slice_loc_low = [
            dcm.SliceLocation for dcm in dict_template_ds_low
        ]

        # Check T2 list is sorted
        expected_list_slice_loc = sorted(
            pydicom.read_file(f, stop_before_pixels=True).SliceLocation
            for f in self.list_filename_in_low
        )
        #
        self.assertEqual(
            list_slice_loc_low,
            expected_list_slice_loc,
            "Low dose Dataset list is not sorted",
        )

    def test_get_dicom_data_from_series(self):
        self.job_obj.task.dict_required_series = {self.zero_series_in.seriesinstanceUID: self.zero_series_in,
                                                  self.low_series_in.seriesinstanceUID: self.low_series_in}
        self.job_obj._get_input_series()
        self._validate_dicom_datasets()

    def test_missing_dicom_data(self):
        '''
        Testing if error is raised on passing a null dicom series.
        '''
        self.job_obj.task.dict_required_series = {'series_found': None}
        with self.assertRaises(Exception):
            self.job_obj._get_input_series()
    
    def test_config_pipeline(self):
        error_config = {"model_type": "gad_process",
        "pipeline_preproc": {
            'gad_process' : {
                'STEP1' : {'op': 'MASK'},
                'STEP2' : {'op': 'SKULLSTRIP'},
                'STEP3' : {'op' : 'REGISTER'},
            }
            
            },
            "pipeline_postproc": {
                'gad_process' : {
                    'STEP1' : {'op' : 'CLIP'},
                    'STEP2' : {'op' : 'RESCALEGLOBAL'}

                }
            }}
        with self.assertRaises(Exception):
            self.job_obj._parse_pipeline_definition(error_config)
    
    def test_default_preprocess(self):

        """
        Testing if the default config steps produce the expected default output
        """

        self.default_pixel_data = self.job_obj._preprocess()

        frame_seq_name = list(self.job_obj._raw_input.keys())[-1]
        #np.save(os.path.join(self.path_data, "default_preprocess.npy"), self.default_pixel_data[frame_seq_name])

        self.default_preprocess_data = np.load(os.path.join(self.path_data, "default_preprocess.npy"))

        self.assertTrue(np.allclose(self.default_pixel_data[frame_seq_name],self.default_preprocess_data, rtol = 100), 'Default Preprocessing is not matching with the expected output')

    
    def test_ge_preprocess(self):
        processing_config = {"model_type": "gad_process",
        "pipeline_preproc": {'gad_process' : {
                'STEP1' : {'op': 'MASK', 'param': {'noise_mask_area': False, 'noise_mask_selem': False}},
                'STEP2' : {'op': 'SKULLSTRIP'},
                'STEP3' : {'op' : 'REGISTER', 'param': {'transform_type': "rigid", 'use_mask_reg': True}},
                'STEP4' : {'op': 'SCALEGLOBAL', 'param': {'joint_normalize': False, 'scale_ref_zero_img': False}}
            }},
            "pipeline_postproc": {
                'gad_process' : {
                    'STEP1' : {'op' : 'CLIP'},
                    'STEP2' : {'op' : 'RESCALEGLOBAL'}

                }}}

        update_exec_config_keys = ['model_type', 'pipeline_preproc', 'pipeline_postproc'
        ]
        self.mock_task.job_definition.exec_config = {
            k: v for k, v in processing_config.items() if k in update_exec_config_keys
        }

        self.job_obj = subtle_gad_jobs.SubtleGADJobType(
            task=self.mock_task, model_dir=self.model_dir
        )

        print(self.job_obj._proc_config.pipeline_preproc)

        self.ge_pixel_data = self.job_obj._preprocess()

        frame_seq_name = list(self.job_obj._raw_input.keys())[-1]
        #np.save(os.path.join(self.path_data, "ge_preprocess.npy"), self.ge_pixel_data[frame_seq_name])

        self.ge_preprocess_data = np.load(os.path.join(self.path_data, "ge_preprocess.npy"))

        self.assertTrue(np.allclose(self.ge_pixel_data[frame_seq_name],self.ge_preprocess_data, rtol = 100), 'GE Preprocessing is not matching with the expected output')


    def test_siemens_preprocess(self):
        processing_config = {"model_type": "gad_process",
        "pipeline_preproc": {'gad_process' : {
                'STEP1' : {'op': 'MASK', 'param': {'noise_mask_area': False, 'noise_mask_selem': False}},
                'STEP2' : {'op': 'SKULLSTRIP'},
                'STEP3' : {'op' : 'REGISTER', 'param': {'transform_type': "rigid", 'use_mask_reg': False}},
                'STEP4' : {'op': 'SCALEGLOBAL', 'param': {'joint_normalize': True, 'scale_ref_zero_img': False}}
            }},
            "pipeline_postproc": {
                'gad_process' : {
                    'STEP1' : {'op' : 'CLIP'},
                    'STEP2' : {'op' : 'RESCALEGLOBAL'}

                }}}

        update_exec_config_keys = ['model_type', 'pipeline_preproc', 'pipeline_postproc'
        ]
        self.mock_task.job_definition.exec_config = {
            k: v for k, v in processing_config.items() if k in update_exec_config_keys
        }

        self.job_obj = subtle_gad_jobs.SubtleGADJobType(
            task=self.mock_task, model_dir=self.model_dir
        )

        self.siemens_pixel_data = self.job_obj._preprocess()

        frame_seq_name = list(self.job_obj._raw_input.keys())[-1]

        self.siemens_preprocess_data = np.load(os.path.join(self.path_data, "siemens_preprocess.npy"))

        self.assertTrue(np.allclose(self.siemens_pixel_data[frame_seq_name],self.siemens_preprocess_data, rtol=100), 'Siemens Preprocessing is not matching with the expected output')

    def test_philips_preprocess(self):
        processing_config = {"model_type": "gad_process",
        "pipeline_preproc": {'gad_process' : {
                'STEP1' : {'op': 'MASK', 'param': {'noise_mask_area': False, 'noise_mask_selem': False}},
                'STEP2' : {'op': 'SKULLSTRIP'},
                'STEP3' : {'op' : 'REGISTER', 'param': {'transform_type': "affine", 'use_mask_reg': False}},
                'STEP4' : {'op' : 'HIST'},
                'STEP5' : {'op': 'SCALEGLOBAL', 'param': {'joint_normalize': False, 'scale_ref_zero_img': False}}
            }},
            "pipeline_postproc": {
                'gad_process' : {
                    'STEP1' : {'op' : 'CLIP'},
                    'STEP2' : {'op' : 'RESCALEGLOBAL'}

                }}}

        update_exec_config_keys = ['model_type', 'pipeline_preproc', 'pipeline_postproc'
        ]
        self.mock_task.job_definition.exec_config = {
            k: v for k, v in processing_config.items() if k in update_exec_config_keys
        }

        self.job_obj = subtle_gad_jobs.SubtleGADJobType(
            task=self.mock_task, model_dir=self.model_dir
        )

        print(self.job_obj._proc_config.pipeline_preproc)

        self.philips_pixel_data = self.job_obj._preprocess()

        frame_seq_name = list(self.job_obj._raw_input.keys())[-1]
        
        self.philips_preprocess_data = np.load(os.path.join(self.path_data, "philips_preprocess.npy"))

        self.assertTrue(np.allclose(self.philips_pixel_data[frame_seq_name],self.philips_preprocess_data, rtol = 100), 'Philips Preprocessing is not matching with the expected output')

    def test_center_crop_even(self):
        """
        Test that center crop result matches the ref image when the input image shape is even
        """

        img = np.ones((7, 256, 256))
        ref_img = np.ones((7, 240, 240))

        crop_img = subtle_gad_jobs.SubtleGADJobType._center_crop(img, ref_img)
        self.assertTrue(crop_img.shape == ref_img.shape)

    def test_center_crop_odd(self):
        """
        Test that center crop result matches the ref image when the input image shape is odd
        """

        img = np.ones((12, 255, 255))
        ref_img = np.ones((7, 240, 240))

        crop_img = subtle_gad_jobs.SubtleGADJobType._center_crop(img, ref_img)
        self.assertTrue(crop_img.shape == ref_img.shape)

    def test_zero_pad(self):
        """
        Test that center crop result matches the ref image when the input image shape is odd
        """

        img = np.ones((7, 232, 248))
        ref_img = np.ones((7, 256, 256))

        pad_img = subtle_gad_jobs.SubtleGADJobType._zero_pad(img, ref_img)
        self.assertTrue(pad_img.shape == ref_img.shape)

    def test_undo_reshape(self):
        """
        Test that undo_reshape function executes correct with the given undo_methods
        """

        undo_methods = [{
            'fn': 'undo_zero_pad',
            'arg': np.zeros((7, 256, 232))
        }, {
            'fn': 'undo_resample',
            'arg': [1.0, 0.5, 0.5]
        }]

        pixel_data = np.zeros((7, 512, 512, 1))

        with mock.patch('subtle_gad_jobs.zoom_interp') as mock_zoom:
            with mock.patch('subtle_gad_jobs.SubtleGADJobType._center_crop') as mock_crop:
                mock_zoom_ret = np.zeros((7, 256, 256))
                mock_zoom.return_value = mock_zoom_ret

                self.job_obj._undo_model_compat_reshape = undo_methods
                self.job_obj._undo_reshape(pixel_data)

                self.assertTrue(mock_zoom.call_count == 1)
                self.assertTrue(np.array_equal(mock_zoom.call_args[0][0], pixel_data[..., 0]))
                self.assertTrue(np.array_equal(mock_zoom.call_args[0][1], undo_methods[1]['arg']))

                self.assertTrue(mock_crop.call_count == 1)
                self.assertTrue(np.array_equal(mock_crop.call_args[0][0], mock_zoom_ret))
                self.assertTrue(np.array_equal(mock_crop.call_args[0][1], undo_methods[0]['arg']))
    
    @pytest.mark.inference
    def test_inference(self):
        try:

            self.default_pixel_data = self.job_obj._preprocess()
            input_dict = self.default_pixel_data
            self.job_obj.mask=[]
            output_array = self.job_obj._process(input_dict)['single_frame']
            #np.save(os.path.join(self.path_data, "pred_output.npy"), output_array)
            self.expected_output_data = np.array(np.load(os.path.join(self.path_data, "pred_output.npy"), allow_pickle = True)).astype(np.float32)

            self.assertTrue(
                np.allclose(output_array, self.expected_output_data, rtol=10),
                "Pixel data is different than ground truth",
            )
        finally:
            if 'job_obj' in locals():
                if hasattr(job_obj._model, 'load_process'):
                    job_obj._model.load_process.terminate()

    @pytest.mark.inference
    def test_postprocess(self):
        self.expected_output_data = np.array(np.load(os.path.join(self.path_data, "pred_output.npy"), allow_pickle = True)).astype(np.float32)
        dict_pixel_data = {}
        dict_pixel_data['single_frame'] = self.expected_output_data
        self.job_obj.global_scale = 774
        self.job_obj._postprocess_data(dict_pixel_data)


if __name__ == "__main__":
    unittest.main()
