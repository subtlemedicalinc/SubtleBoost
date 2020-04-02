"""
Unit test for preprocessing in SubtleGad jobs

@author: Srivathsa Pasumarthi <srivathsa@subtlemedical.com>
Copyright Subtle Medical, Inc. (https://subtlemedical.com)
Created on 2020/03/02
"""

import os
import unittest
from unittest.mock import MagicMock
import pytest
import mock
import numpy as np

from subtle.dcmutil.series_utils import dicomscan
# pylint: disable=import-error
import subtle_gad_jobs


@pytest.mark.processing
class PreprocessTest(unittest.TestCase):
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
            "use_mask_reg": False
        }

        path_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
        self.model_dir = os.path.join(path_data, "model", "simple_test")

        self.processing_config = processing_config

        self.mock_task = MagicMock()
        self.mock_task.job_name = 'test'
        self.mock_task.job_definition.exec_config = self.processing_config

        self.job_obj = subtle_gad_jobs.SubtleGADJobType(
            task=self.mock_task, model_dir=self.model_dir
        )

        self.zero_path_dicom = os.path.join(path_data, "NO26_Philips_Gad_small", "0_zerodose")
        self.zero_series_in = list(dicomscan(self.zero_path_dicom).values())[0]

        self.low_path_dicom = os.path.join(path_data, "NO26_Philips_Gad_small", "1_lowdose")
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

        self.dummy_data = np.ones((2, 7, 240, 240))

    def test_mask_noise_skip(self):
        """
        Test to make sure that noise masking is skipped when the app is configured with
        `perform_noise_mask=False`
        """

        self.processing_config['perform_noise_mask'] = False
        self.job_obj._proc_config = self.job_obj.SubtleGADProcConfig(**self.processing_config)

        _, mask = self.job_obj._mask_noise(self.dummy_data)
        self.assertTrue(np.array_equal(mask, np.array([])))

    def test_mask_noise_exec(self):
        """
        When the app is configured with `perform_noise_mask=True`, make sure that the corresponding
        app utilities function is called with the correct arguments
        """

        with mock.patch('subtle_gad_jobs.preprocess_single_series') as mock_pre:
            mock_pre.mask_bg_noise.return_value = np.zeros((7, 240, 240))

            self.job_obj._mask_noise(self.dummy_data)
            self.assertTrue(mock_pre.mask_bg_noise.call_count == 2)

            self.assertTrue(
                np.array_equal(mock_pre.mask_bg_noise.call_args[1]['img'], np.zeros((7, 240, 240)))
            )

            self.assertEqual(
                mock_pre.mask_bg_noise.call_args[1]['threshold'],
                self.job_obj._proc_config.noise_mask_threshold
            )

            self.assertEqual(self.job_obj._proc_lambdas[-1]['name'], 'noise_mask')

    def test_skull_strip_npy(self):
        """
        Test that skull strip npy function calls the corresponding deepbrain and app
        utilities function
        """

        with mock.patch('subtle_gad_jobs.BrainExtractor') as mock_bext:
            with mock.patch('subtle_gad_jobs.np') as mock_np:
                with mock.patch('subtle_gad_jobs.preprocess_single_series') as mock_pre:
                    bext_obj = MagicMock()
                    mock_bext.return_value = bext_obj
                    mock_np.interp.return_value = self.dummy_data
                    bext_obj.run.return_value = np.copy(self.dummy_data)

                    self.job_obj._strip_skull_npy_vol(self.dummy_data)

                    mock_bext.assert_called_once_with()
                    bext_obj.run.assert_called_once_with(self.dummy_data)
                    self.assertTrue(mock_pre.get_largest_connected_component.call_count == 1)
                    self.assertTrue(np.array_equal(
                        mock_pre.get_largest_connected_component.call_args[0][0],
                        np.array(self.dummy_data, dtype=bool)
                    ))

    def test_skull_strip_disable(self):
        """
        Test that skull stripping is not performed when disabled in proc config
        """

        mock_fn = MagicMock()
        self.job_obj._strip_skull_npy_vol = mock_fn

        self.processing_config['skull_strip'] = False
        self.job_obj._proc_config = self.job_obj.SubtleGADProcConfig(**self.processing_config)

        self.job_obj._strip_skull(self.dummy_data)
        self.assertTrue(mock_fn.call_count == 0)

    def test_skull_strip_union_disable(self):
        """
        Test that only one skull strip operation is performed when `skull_strip_union=False`
        """

        mock_fn = MagicMock()
        self.job_obj._strip_skull_npy_vol = mock_fn

        self.processing_config['skull_strip_union'] = False
        self.job_obj._proc_config = self.job_obj.SubtleGADProcConfig(**self.processing_config)

        self.job_obj._strip_skull(self.dummy_data)
        self.assertTrue(mock_fn.call_count == 1)
        self.assertTrue(np.array_equal(mock_fn.call_args[0][0], self.dummy_data[0]))

    def test_skull_strip_all(self):
        """
        Test that skull strip operation is called twice when `skull_strip_union=True`
        """

        mock_fn = MagicMock()
        self.job_obj._strip_skull_npy_vol = mock_fn
        mock_fn.return_value = np.copy(self.dummy_data[0])

        self.job_obj._strip_skull(self.dummy_data)
        self.assertTrue(mock_fn.call_count == 2)

    def test_scale_slope(self):
        """
        Test that input image is properly scaled with the given slope and intercept values
        """

        img_scale = subtle_gad_jobs.SubtleGADJobType._scale_slope_intercept(
            self.dummy_data, rescale_slope=2.0, rescale_intercept=3.0, scale_slope=2.0
        )
        self.assertTrue(np.array_equal(img_scale, self.dummy_data * 1.25))

    def test_rescale_slope(self):
        """
        Test that input image is properly rescaled with the given slope and intercept values
        """

        img_rescale = subtle_gad_jobs.SubtleGADJobType._rescale_slope_intercept(
            self.dummy_data, rescale_slope=2.0, rescale_intercept=3.0, scale_slope=2.0
        )
        self.assertTrue(np.array_equal(img_rescale, self.dummy_data * 0.5))

    def test_scale_dicom_tags_disable(self):
        """
        Test that dicom scaling is not performed when proc config has `perform_dicom_scaling=False`
        """

        try:
            original_fn = subtle_gad_jobs.SubtleGADJobType._scale_slope_intercept
            mock_fn = MagicMock()
            subtle_gad_jobs.SubtleGADJobType._scale_slope_intercept = mock_fn

            self.job_obj._scale_intensity_with_dicom_tags(self.dummy_data)
            self.assertTrue(mock_fn.call_count == 0)
        finally:
            subtle_gad_jobs.SubtleGADJobType._scale_slope_intercept = original_fn

    def test_scale_dicom_tags_enable(self):
        """
        Test that the corresponding functions are called and dicom scaling is performed when
        proc config has `perform_dicom_scaling=True`
        """

        try:
            self.processing_config['perform_dicom_scaling'] = True
            self.job_obj._proc_config = self.job_obj.SubtleGADProcConfig(**self.processing_config)

            original_fn = subtle_gad_jobs.SubtleGADJobType._scale_slope_intercept
            mock_fn = MagicMock()
            mock_fn.return_value = np.copy(self.dummy_data[0])
            subtle_gad_jobs.SubtleGADJobType._scale_slope_intercept = mock_fn

            self.job_obj._scale_intensity_with_dicom_tags(self.dummy_data)

            self.assertTrue(mock_fn.call_count == 2)
            self.assertTrue(self.job_obj._proc_lambdas[-1]['name'], 'dicom_scaling')
            self.assertTrue(self.job_obj._undo_dicom_scale_fn is not None)
        finally:
            subtle_gad_jobs.SubtleGADJobType._scale_slope_intercept = original_fn

    def test_register(self):
        """
        Test that app utilities preprocess module's registration function is called with the
        appropriate arguments
        """

        with mock.patch('subtle_gad_jobs.preprocess_single_series') as mock_pre:
            mock_pre.register = MagicMock()
            mock_pre.register.return_value = (np.copy(self.dummy_data[0]), MagicMock())

            zeros = np.zeros((7, 240, 240))
            ones = np.ones((7, 240, 240))
            reg_input = np.array([zeros, ones])
            self.job_obj._register(reg_input)

            self.assertTrue(mock_pre.register.call_count == 1)

            call_args = mock_pre.register.call_args[1]
            self.assertTrue(np.array_equal(call_args['fixed_img'], zeros))
            self.assertTrue(np.array_equal(call_args['moving_img'], ones))
            self.assertTrue(call_args['fixed_spacing'] == self.job_obj._pixel_spacing[0])
            self.assertTrue(call_args['moving_spacing'] == self.job_obj._pixel_spacing[1])
            self.assertTrue(
                call_args['transform_type'] == self.job_obj._proc_config.transform_type
            )

            self.assertTrue(self.job_obj._proc_lambdas[-1]['name'], 'register')

    def test_match_hist_disable(self):
        """
        Test that histogram matching is not performed when proc config has
        `histogram_matching=False`
        """

        with mock.patch('subtle_gad_jobs.preprocess_single_series') as mock_pre:
            self.processing_config['histogram_matching'] = False
            self.job_obj._proc_config = self.job_obj.SubtleGADProcConfig(**self.processing_config)
            mock_pre.match_histogram = MagicMock()

            self.job_obj._match_histogram(self.dummy_data)
            self.assertTrue(mock_pre.match_histogram.call_count == 0)

    def test_match_hist_enable(self):
        """
        Test that histogram matching method from app utilities is called with appropriate
        arguments when proc config has `histogram_matching=True`
        """

        with mock.patch('subtle_gad_jobs.preprocess_single_series') as mock_pre:
            mock_pre.match_histogram = MagicMock()
            mock_pre.match_histogram.return_value = np.copy(self.dummy_data[0])

            zeros = np.zeros((7, 240, 240))
            ones = np.ones((7, 240, 240))
            hist_input = np.array([zeros, ones])
            self.job_obj._match_histogram(hist_input)

            self.assertTrue(mock_pre.match_histogram.call_count == 1)
            self.assertTrue(np.array_equal(mock_pre.match_histogram.call_args[1]['img'], ones))
            self.assertTrue(
                np.array_equal(mock_pre.match_histogram.call_args[1]['ref_img'], zeros)
            )
            self.assertTrue(self.job_obj._proc_lambdas[-1]['name'], 'histogram_matching')

    def test_scale_intensity(self):
        """
        Test that get scale intensity function from app utilities is called with appropriate
        arguments
        """

        with mock.patch('subtle_gad_jobs.preprocess_single_series') as mock_pre:
            self.processing_config['num_scale_context_slices'] = 7
            self.job_obj._proc_config = self.job_obj.SubtleGADProcConfig(**self.processing_config)

            mock_pre.get_intensity_scale = MagicMock()
            mock_pre.get_intensity_scale.return_value = 0.9

            zeros = np.zeros((7, 240, 240))
            ones = np.ones((7, 240, 240))
            scale_input = np.array([zeros, ones])

            self.job_obj._scale_intensity(scale_input, self.dummy_data)
            self.assertTrue(mock_pre.get_intensity_scale.call_count == 1)
            self.assertTrue(np.array_equal(
                mock_pre.get_intensity_scale.call_args[1]['levels'], np.linspace(0.5, 1.5, 30)
            ))

            self.assertTrue(self.job_obj._proc_lambdas[-1]['name'], 'global_scale')
            self.assertTrue(self.job_obj._proc_lambdas[-2]['name'], 'match_scales')
            self.assertTrue(self.job_obj._undo_global_scale_fn is not None)

    def test_apply_proc_lambdas(self):
        """
        Test that _apply_proc_lambdas function applies the given lambda functions on the given
        unmasked data
        """

        ml1 = MagicMock()
        ml1.return_value = np.copy(self.dummy_data[0])

        ml2 = MagicMock()
        ml2.return_value = np.copy(self.dummy_data[0])

        proc_lambdas = [{
            'name': 'mock_lambda1',
            'fn': [ml1]
        }, {
            'name': 'mock_lambda2',
            'fn': [ml2]
        }]

        self.job_obj._proc_lambdas = proc_lambdas
        self.job_obj._apply_proc_lambdas(self.dummy_data)

        self.assertTrue(ml1.call_count == 1)
        self.assertTrue(np.array_equal(ml1.call_args[0][0], self.dummy_data))

        self.assertTrue(ml2.call_count == 1)
        self.assertTrue(np.array_equal(ml2.call_args[0][0], self.dummy_data))

if __name__ == "__main__":
    unittest.main()
