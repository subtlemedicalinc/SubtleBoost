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
import mock
import numpy as np
import pydicom

from subtle.dcmutil.series_utils import dicomscan
# pylint: disable=import-error
import subtle_gad_jobs

@pytest.mark.processing
class InferenceTest(unittest.TestCase):
    """
    A unittest framework to test the preprocessing functions
    """

    def setUp(self):
        """
        Setup proc config and init params for job class
        """
        processing_config = {
            "app_name": "SubtleGAD",
            "app_id": 3000,
            "app_version": "unittest",
            "model_id": "None",
            "not_for_clinical_use": False,
            "perform_noise_mask": True,
            "noise_mask_threshold": 0.1,
            "noise_mask_area": False,
            "noise_mask_selem": False,
            "perform_dicom_scaling": False,
            "transform_type": "rigid",
            "histogram_matching": False,
            "joint_normalize": False,
            "scale_ref_zero_img": False,
            "skull_strip": True,
            "skull_strip_union": True,
            "skull_strip_prob_threshold": 0.5,
            "num_scale_context_slices": 20,
            "inference_mpr": True,
            "num_rotations": 5,
            "slices_per_input": 7,
            "mpr_angle_start": 0,
            "mpr_angle_end": 90,
            "reshape_for_mpr_rotate": True,
            "num_procs_per_gpu": 2,
            "series_desc_prefix": "SubtleGAD:",
            "series_desc_suffix": "",
            "series_number_offset": 100,
            "use_mask_reg": False,
            "acq_plane": "AX",
            "blur_lowdose": False,
            "model_resolution": [1.0, 0.5, 0.5],
            "perform_registration": True,
            "min_gpu_mem_mb": 9800.0,
            "cs_blur_sigma": [0, 1.5]
        }

        self.path_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")

        self.model_dir = os.path.join(self.path_data, "model", "simple_test")

        self.processing_config = processing_config

        self.mock_task = MagicMock()
        self.mock_task.job_name = 'test'
        exec_config_keys = [
            'app_name', 'app_id', 'app_version', 'model_id', 'not_for_clinical_use',
            'series_desc_prefix', 'series_desc_suffix', 'series_number_offset'
        ]
        self.mock_task.job_definition.exec_config = {
            k: v for k, v in processing_config.items() if k in exec_config_keys
        }

        self.job_obj = subtle_gad_jobs.SubtleGADJobType(
            task=self.mock_task, model_dir=self.model_dir
        )

        self.zero_path_dicom = os.path.join(self.path_data, "NO26_Philips_Gad_small", "0_zerodose")
        self.zero_series_in = list(dicomscan(self.zero_path_dicom).values())[0]

        self.low_path_dicom = os.path.join(self.path_data, "NO26_Philips_Gad_small", "1_lowdose")
        self.low_series_in = list(dicomscan(self.low_path_dicom).values())[0]

        self.sequence_name = self.zero_series_in.get_frame_names()[0]

        self.job_obj.task.dict_required_series = {
            'zero_dose_philips': self.zero_series_in,
            'low_dose_philips': self.low_series_in
        }

        self.job_obj._get_dicom_data()
        self.job_obj._set_mfr_specific_config()
        self.job_obj._get_pixel_spacing()
        self.job_obj._get_dicom_scale_coeff()
        self.job_obj._get_raw_pixel_data()

    def test_process_mpr(self):
        """
        Test that one single instance of MPR inference is executed properly by testing the
        sample test model's prediction against the expected prediction
        """

        mock_pool = MagicMock()
        mock_pool.get.return_value = "0"
        subtle_gad_jobs._init_gpu_pool(mock_pool)

        mpr_params = {
            'model_dir': self.model_dir,
            'decrypt_key_hex': None,
            'exec_config': self.processing_config,
            'slices_per_input': self.job_obj._proc_config.slices_per_input,
            'reshape_for_mpr_rotate': self.job_obj._proc_config.reshape_for_mpr_rotate,
            'data': self.job_obj._raw_input[self.sequence_name],
            'angle': 45.0,
            'slice_axis': 1
        }

        expected_pred = np.load(os.path.join(self.path_data, "expected_pred.npy"))
        result = subtle_gad_jobs.SubtleGADJobType._process_mpr(mpr_params)

        self.assertTrue(np.allclose(result, expected_pred, atol=1.0))
        mock_pool.get.assert_called_once_with(block=True)
        mock_pool.put.assert_called_once_with("0")

    def test_compat_reshape(self):
        """
        Test the _process_model_input_compatibility method
        """

        with mock.patch('subtle_gad_jobs.GenericInferenceModel') as mock_model:
            with mock.patch('subtle_gad_jobs.SubtleGADJobType._zero_pad') as mock_pad:
                with mock.patch('subtle_gad_jobs.zoom_interp') as mock_interp:
                    mock_inf_model = MagicMock()
                    mock_inf_model._model_obj.inputs = [np.zeros((14, 512, 512))]
                    mock_model.return_value = mock_inf_model
                    self.job_obj._pixel_spacing = [[0.75, 0.75, 1.0]]
                    mock_pad.return_value = np.zeros((2, 7, 256, 256))
                    mock_interp.return_value = np.zeros((7, 384, 384))

                    _, undo_methods = self.job_obj._process_model_input_compatibility(
                        np.zeros((2, 7, 232, 256))
                    )

                    self.assertTrue(mock_pad.call_count == 2)
                    self.assertTrue(mock_interp.call_count == 2)

                    undo_fn_names = [m['fn'] for m in undo_methods]
                    self.assertTrue(len(undo_methods) == 3)
                    self.assertTrue(
                        np.array_equal(
                            undo_fn_names, ['undo_zero_pad', 'undo_resample', 'undo_zero_pad']
                        )
                    )

    def test_gpu_allocation(self):
        """
        Test the _get_available_gpus method
        """

        with mock.patch('subtle_gad_jobs.GPUtil') as mock_stats:
            gpu_obj = namedtuple('gpu_obj', 'id memoryTotal memoryUsed')
            mock_stats.getGPUs.return_value = [
                gpu_obj(id=0, memoryTotal=10989, memoryUsed=4500),
                gpu_obj(id=1, memoryTotal=10989, memoryUsed=989),
                gpu_obj(id=2, memoryTotal=10989, memoryUsed=0),
                gpu_obj(id=3, memoryTotal=10989, memoryUsed=1189)
            ]

            gpus = self.job_obj._get_available_gpus()
            self.assertTrue(gpus == '1,2,3')

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

    def test_postprocess(self):
        """
        Test that postprocess is executed correctly by checking that the undo lambda functions are
        called with appropriate arguments
        """

        with mock.patch('subtle_gad_jobs.np') as mock_np:
            self.processing_config['perform_dicom_scaling'] = True
            self.job_obj._proc_config = \
            self.job_obj.SubtleGADProcConfig(**self.processing_config)

            dummy_data = np.ones((2, 7, 240, 240))
            self.job_obj._undo_global_scale_fn = MagicMock()
            self.job_obj._undo_global_scale_fn.return_value = dummy_data

            self.job_obj._undo_dicom_scale_fn = MagicMock()
            self.job_obj._undo_dicom_scale_fn.return_value = dummy_data
            mock_np.clip.return_value = dummy_data

            self.job_obj._postprocess_data({self.sequence_name: dummy_data})
            self.job_obj._undo_global_scale_fn.assert_called_once_with(dummy_data)
            self.job_obj._undo_dicom_scale_fn.assert_called_once_with(dummy_data)
            self.assertTrue(
                np.array_equal(self.job_obj._output_data[self.sequence_name], dummy_data)
            )

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

    def test_save_data(self):
        """
        Test save dicom data function by checking that the destination directory has the expected
        number of DICOM files and that the DICOM file's pixel_data is appropriate
        """

        out_dir = tempfile.mkdtemp()

        dummy_data = np.ones((7, 240, 240, 1))
        self.job_obj._save_data(
            dict_pixel_data={self.sequence_name: dummy_data},
            dict_template_ds=self.job_obj._input_datasets[0],
            out_dicom_dir=out_dir
        )

        dcm_files = glob('{}/**/*.dcm'.format(out_dir), recursive=True)
        self.assertTrue(len(dcm_files) == 7)
        self.assertTrue(
            list(self.job_obj._input_datasets[0].values())[0][0].StudyInstanceUID in dcm_files[0]
        )

        dcm_pixel_array = pydicom.dcmread(dcm_files[0]).pixel_array
        self.assertTrue(np.array_equal(dcm_pixel_array, np.ones((240, 240))))

        shutil.rmtree(out_dir)

if __name__ == "__main__":
    unittest.main()
