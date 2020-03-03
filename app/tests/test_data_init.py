"""
Unit test for data init in SubtleGad jobs

@author: Srivathsa Pasumarthi <srivathsa@subtlemedical.com>
Copyright Subtle Medical, Inc. (https://subtlemedical.com)
Created on 2020/02/27
"""

import os
import unittest
from unittest.mock import MagicMock
import pytest
import numpy as np
# pylint: disable=import-error
from subtle_gad_jobs import SubtleGADJobType
from subtle.dcmutil.series_utils import dicomscan

@pytest.mark.processing
class DataLoadingTest(unittest.TestCase):
    """
    A unittest framework to test the data loading logic in SubtleGAD job
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
            "series_number_offset": 100
        }

        path_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
        self.model_dir = os.path.join(path_data, "model", "simple_test")

        self.mock_task = MagicMock()
        self.mock_task.job_name = 'test'
        self.mock_task.job_definition.exec_config = processing_config

        self.job_obj = SubtleGADJobType(task=self.mock_task, model_dir=self.model_dir)

        self.zero_path_dicom = os.path.join(path_data, "NO26_Philips_Gad_small", "0_zerodose")
        self.zero_series_in = list(dicomscan(self.zero_path_dicom).values())[0]

        self.low_path_dicom = os.path.join(path_data, "NO26_Philips_Gad_small", "1_lowdose")
        self.low_series_in = list(dicomscan(self.low_path_dicom).values())[0]

        self.sequence_name = self.zero_series_in._default_frame_name

        self.job_obj.task.dict_required_series = {
            'zero_dose_philips': self.zero_series_in,
            'low_dose_philips': self.low_series_in
        }

    def test_get_dicom_assert_len(self):
        """
        Test to make sure _get_dicom_data raises assertion error when there are not
        enough number of series defined in the input task
        """

        with self.assertRaises(AssertionError):
            self.job_obj.task.dict_required_series = {
                'zero_dose_philips': self.zero_series_in
            }

            self.job_obj._get_dicom_data()

    def test_get_dicom_data(self):
        """
        Test to check the correct execution of _get_dicom_data by making sure the
        corresponding series instance UIDs match
        """

        self.job_obj._get_dicom_data()
        self.assertEqual(
            self.job_obj._input_datasets[0][self.sequence_name][0].SeriesInstanceUID,
            self.zero_series_in.seriesinstanceUID
        )

        self.assertEqual(
            self.job_obj._input_datasets[1][self.sequence_name][0].SeriesInstanceUID,
            self.low_series_in.seriesinstanceUID
        )

    def test_mfr_config_errors(self):
        """
        Test that _set_mfr_specific_config method raises errors when manufacturer tag is not
        present or tags do not match between zero and low dose series
        """

        self.job_obj._get_dicom_data()

        with self.assertRaises(TypeError):
            self.job_obj._input_series[0].manufacturer = None
            self.job_obj._set_mfr_specific_config()

        with self.assertRaises(TypeError):
            self.job_obj._input_series[0].manufacturer = 'philips2'
            self.job_obj._set_mfr_specific_config()

    def test_mfr_config_no_matching_mfr(self):
        """
        Test that _set_mfr_specific_config sets default config when the input series manufacturer
        is not one among the defined manufacturers
        """

        self.job_obj._get_dicom_data()

        self.job_obj._input_series[0].manufacturer = 'hitachi'
        self.job_obj._input_series[1].manufacturer = 'hitachi'
        self.job_obj._set_mfr_specific_config()

        self.assertEqual(self.job_obj._proc_config.noise_mask_threshold, 0.05)

    def test_mfr_config_exec(self):
        """
        Test that _set_mfr_specific_config sets 'philips' manufacturer for the given input series
        """

        self.job_obj._get_dicom_data()
        self.job_obj._set_mfr_specific_config()
        self.assertEqual(self.job_obj._proc_config.noise_mask_threshold, 0.08)

    def test_get_pixel_spacing(self):
        """
        Test that the _get_pixel_spacing method assigns the correct pixel spacing to the job
        object class
        """

        self.job_obj._get_dicom_data()
        self.job_obj._get_pixel_spacing()
        zero_ps, low_ps = self.job_obj._pixel_spacing
        zero_ps = [round(float(p), 3) for p in zero_ps]
        low_ps = [round(float(p), 3) for p in low_ps]

        self.assertEqual(zero_ps, [1.0, 1.0, 1.0])
        self.assertEqual(low_ps, [1.0, 1.0, 1.0])

    def test_dicom_coeff_default(self):
        """
        Test that _get_dicom_scale_coeff returns default values when scaling information is not
        available from DICOM header
        """

        self.job_obj._get_dicom_data()
        del self.job_obj._input_datasets[0][self.sequence_name][0].RescaleSlope
        self.job_obj._get_dicom_scale_coeff()

        assert round(self.job_obj._dicom_scale_coeff[0]['rescale_slope'], 2) == 1.0

    def test_dicom_coeff_values(self):
        """
        Test that _get_dicom_scale_coeff gets the correct dicom scaling coefficients from header
        """

        self.job_obj._get_dicom_data()
        self.job_obj._get_dicom_scale_coeff()

        self.assertEqual(round(self.job_obj._dicom_scale_coeff[0]['rescale_slope'], 2), 4.18)
        self.assertEqual(round(self.job_obj._dicom_scale_coeff[1]['rescale_slope'], 2), 3.29)

    def test_raw_pixel_data(self):
        """
        Test _get_raw_pixel_data method to make sure that the raw pixel data is set correctly
        """

        self.job_obj._get_dicom_data()
        self.job_obj._get_raw_pixel_data()

        self.assertEqual(self.job_obj._raw_input[self.sequence_name].shape, (2, 7, 240, 240))
        self.assertTrue(np.array_equal(
            self.job_obj._raw_input[self.sequence_name][0],
            self.job_obj._input_series[0].get_pixel_data()[self.sequence_name]
        ))

if __name__ == "__main__":
    unittest.main()
