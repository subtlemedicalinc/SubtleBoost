import copy
from datetime import date, timedelta, datetime
import os
import json
import shutil
import subprocess
import re
import time
import logging
import tempfile
from typing import Optional
import unittest
import glob
import numpy as np
import pydicom
import pytest
import yaml
import ismrmrd
import multiprocessing
from collections import OrderedDict
from subtle.dcmutil.series_io import dicomscan
from subtle.testutil.post_build_utils import MainPostBuild
from subtle.util.licensing import generate_license


class PostBuildTest(MainPostBuild):
    """
    A unittest framework to test the inference pipeline
    """

    SERIAL_NUMBER = "526AE600-62B9-11E9-93C8-DF4923B24457"

    def _conform_version(self, config_file: str):
        """
        Make sure software version is the same as that of the config file

        :param config_file: the file to be changed
        """
        with open(os.path.join(self.build_dir, "manifest.json"), "r") as fp:
            manifest = json.load(fp)
        with open(os.path.join(self.build_dir, "config.yml"), "r") as fp:
            default_config = yaml.safe_load(fp)
        with open(config_file, "r") as fp:
            config_dict = yaml.safe_load(fp)
        config_dict["version"] = manifest["version"]
        for job in config_dict['jobs']:
            default_jobs = {j["job_name"]: j for j in default_config["jobs"]}
            assert job["job_name"] in default_jobs
            job['exec_config'] = default_jobs[job["job_name"]]["exec_config"]
        with open(config_file, "w") as fp:
            yaml.safe_dump(config_dict, fp)

    @classmethod
    def setUp(cls,self):
        # model files
        self.test_id = "SubtleGad-" + os.environ.get("POST_TEST_TAG", "UNKNOWN")
        this_dir = os.path.dirname((__file__))

        self.path_data = os.path.join(
            os.path.abspath(this_dir), "post_build_test_data"
        )
        self.build_dir = os.path.abspath(os.path.join(this_dir, "../../dist"))

        super(PostBuildTest, cls).setUp(self, self.path_data,self.build_dir)

        print('path data', self.path_data)

        #self.tmp_folder = tempfile.mkdtemp(dir=self.path_data)
        #self.output_folder = os.path.join(self.tmp_folder, "output")

        # input and output data
        # list of tuple (data_folder, roi_csv)
        self.list_input = [os.path.join(self.path_data, "Gad_test")]
        self.input_folder = os.path.join(self.path_data, "Gad_test")

        #Zero dose and low dose dataset
        self.input_folder_zero = os.path.join(self.path_data, "Gad_test/OSag_3D_T1BRAVO_zero_dose_7/")
        self.input_folder_low = os.path.join(self.path_data, "Gad_test/OSag_3D_T1BRAVO_low_dose_8/")

        #PHI Free Dataset
        self.input_folder_no_PHI = os.path.join(self.path_data, "Gad_test_PHI")

        #Register Failure Dataset
        self.input_register_failure_folder = os.path.join(self.path_data, "register_failure")

        #ZD > LD data
        self.input_zd_dicom = os.path.join(self.path_data,"zd_g_ld_data", "input")
        self.output_zd_dicom = os.path.join(self.path_data, "zd_g_ld_data", "output")


        #LD> ZD data
        self.input_ld_dicom = os.path.join(self.path_data, "ld_g_zd_data", "input")
        self.output_ld_dicom = os.path.join(self.path_data, "ld_g_zd_data", "output")

        #Config files
        self.config_file = os.path.join(self.build_dir, "config.yml")

        #Output Folder
        self.tmp_folder = tempfile.mkdtemp(dir=self.path_data)
        self.output_folder = os.path.join(self.tmp_folder, "output")

        global output_f
        output_f = self.output_folder

        #License File
        license_info = generate_license(
            4000,
            "SubtleGAD",
            self.SERIAL_NUMBER,
            date.today() + timedelta(days=2),
        )

        self.license_file = os.path.join(self.path_data, "license.json")
        # write license
        with open(self.license_file, "w") as f:
            json.dump(license_info, f)

        #self.inferen()
        self.config_copy = os.path.join(self.build_dir, "copyconfig.yml")
            
    
    def inferen(self):
        self.output_inference = self.pre_test("UN5", self.output_folder)
        self.completed_process = self.run_inference(
            self.input_folder,
            self.output_inference,
            self.config_file,
            self.license_file,
        )

    def run_inference(
        self,
        input_folder,
        output_folder,
        config_file,
        license_file,
        cuda_device="0",
    ):

        interface_script = "run.sh"

        cmd_str = (
            "export CUDA_VISIBLE_DEVICES='{cuda}'; "
            "{interface} {input_folder} {output_folder} "
            "{config} {license}"
        )

        cmd = cmd_str.format(
            cuda=cuda_device,
            interface=os.path.join(self.build_dir, interface_script),
            input_folder=input_folder,
            output_folder=output_folder,
            config=config_file,
            license=license_file,
        )

        completed_process = subprocess.run(cmd,
                                           cwd=self.build_dir,
                                           shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

        logging.info(completed_process.stdout.decode())
        logging.info(completed_process.stderr.decode())

        return completed_process
    
    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_dicom_in(self):
        """
        test the infer script of SubtleGad completes execution
        """
        MainPostBuild.t_dicom_in(self)

    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_no_negative_in_output(self):
        """
        test that no negative pixel value is generated
        """
        MainPostBuild.t_no_negative_in_output(self)

    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_input_args(self):
        """
        REQ-32: SubtleApp shall be able to operate on PHI-free DICOM inputs.
        """
        MainPostBuild.t_input_args(self)
    
    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_validate_config(self):
        """
        REQ-5: SubtleApp shall take in Configuration File to specify processing parameters
        """
        MainPostBuild.t_validate_config(self)

    
    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_phi_free(self):
        """
            REQ-32: SubtleApp shall be able to operate on PHI-free DICOM inputs.
        """
        MainPostBuild.t_phi_free(self)

    
    # @pytest.mark.post_build
    # @pytest.mark.subtleapp
    # def test_output_different_series_uid(self):
    #     """
    #     REQ-20: SubtleApp shall update series instance for output image to ‘Processed by SubtleApp’ and
    #     assign a unique new SeriesInstanceUID for output image.
    #     """
    #     MainPostBuild.t_output_different_series_uid(self)
    
    # @pytest.mark.post_build
    # @pytest.mark.subtleapp
    # def test_output_different_series_description(self):
    #     """
    #     REQ-20: SubtleApp shall update series instance for output image to ‘Processed by SubtleApp’ and
    #     assign a unique new SeriesDescription for output image.
    #     """
    #     MainPostBuild.t_output_different_series_description(self)

    
    # @pytest.mark.req38
    # @pytest.mark.ver27
    # @pytest.mark.post_build
    # @pytest.mark.enhanced_dicom
    # def test_enhanced_dicom(self):
    #     """
    #     REQ-38 : SubtleGad shall operate on enhanced DICOMs. 
    #     """
    #     return MainPostBuild.enhanced_dicom(self)

    # @pytest.mark.req36
    # @pytest.mark.ver25
    # @pytest.mark.post_build
    # @pytest.mark.inference
    # @pytest.mark.minimal_test
    # def test_compressed_dicom(self):
    #     """
    #     REQ-36: SubtleGad shall operate on compression based dicoms. Test that compressed dicom studies are read, processed, and output saved
    #     """    

    #     return MainPostBuild.compressed_dicom_processing(self)
    
    # @pytest.mark.req37
    # @pytest.mark.ver26
    # @pytest.mark.post_build
    # @pytest.mark.inference
    # @pytest.mark.register_fail
    # def test_register_failure(self):
    #     """
    #     REQ-37: SubtleGAD shall provide an error message when registration fails. 
    #     """
    #     return MainPostBuild.t_register_failure(self)

    # @pytest.mark.req34
    # @pytest.mark.ver16
    # @pytest.mark.post_build
    # @pytest.mark.inference
    # def test_reject_license(self):
    #     """
    #     REQ-6: SubtleGad shall have CLI detailed the attachment
    #     """
    #     # run test
    #     output_folder = self.pre_test("REQ16")

    #     # create license
    #     input_folder_license = os.path.join(self.tmp_folder, "input_license")
    #     os.makedirs(input_folder_license, exist_ok=True)

    #     # empty
    #     license_info = {"SerialNumber": "", "Expiration": "", "LicenseKey": ""}
    #     license_file = os.path.join(input_folder_license, "license_empty.json")
    #     # write license
    #     with open(license_file, "w") as f:
    #         json.dump(license_info, f)
    #     completed_process = self.run_inference(
    #         self.input_folder,
    #         output_folder,
    #         self.config_file_small,
    #         license_file,
    #     )
    #     # check that execution failed correctly
    #     self.validate_secondary_capture_generic(
    #         90, output_folder, completed_process
    #     )

        
    #     # wrong key
    #     license_info = generate_license(
    #         6000,
    #         "SubtleGAD",
    #         self.SERIAL_NUMBER,
    #         date.today() + timedelta(days=2),
    #     )
    #     license_info["LicenseKey"] = "xxx"
    #     license_file = os.path.join(
    #         input_folder_license, "license_wrong_key.json"
    #     )
    #     # write license
    #     with open(license_file, "w") as f:
    #         json.dump(license_info, f)
    #     completed_process = self.run_inference(
    #         self.input_folder,
    #         output_folder,
    #         self.config_file_small,
    #         license_file,
    #     )
    #     # check that execution failed correctly
    #     self.validate_secondary_capture_generic(
    #         90, output_folder, completed_process
    #     )

        
    #     # expired
    #     license_info = generate_license(
    #         6000,
    #         "SubtleGAD",
    #         self.SERIAL_NUMBER,
    #         date.today() - timedelta(days=2),
    #     )
    #     license_file = os.path.join(
    #         input_folder_license, "license_expired.json"
    #     )
    #     # write license
    #     with open(license_file, "w") as f:
    #         json.dump(license_info, f)
    #     completed_process = self.run_inference(
    #         self.input_folder,
    #         output_folder,
    #         self.config_file_small,
    #         license_file,
    #     )
    #     # check that execution failed correctly
    #     self.validate_secondary_capture_generic(
    #         90, output_folder, completed_process
    #     )

    #     # remove output folder
    #     shutil.rmtree(output_folder)
    #     shutil.rmtree(input_folder_license)

    # def validate_secondary_capture_generic(
    #     self, expected_exit_code, output_folder, completed_process
    # ):
    #     # check that execution failed with NoMatchJobs Exception
    #     self.assertEqual(
    #         completed_process.returncode,
    #         expected_exit_code,
    #         msg="Execution didn't fail as expected: {}".format(
    #             completed_process.args
    #         ),
    #     )

    #     # check that the error report folder exists
    #     error_report_folder = os.path.join(output_folder, "dicoms", "error_report")
    #     self.assertTrue(os.path.isdir(error_report_folder))
    #     self.assertGreater(len(os.listdir(error_report_folder)), 0)

    #     # check that the error report was output correctly
    #     error_report_file = glob.glob(os.path.join(error_report_folder,
    #                                                "ERR_{}_000000*.dcm".format(expected_exit_code)
    #                                                )
    #                                   )[0]
    #     self.assertTrue(
    #         os.path.isfile(error_report_file), msg="Error report file not found"
    #     )

    #     # check that error report has the correct SOPClassUID
    #     ds = pydicom.read_file(error_report_file)
    #     self.assertEqual(
    #         ds.SOPClassUID,
    #         "1.2.840.10008.5.1.4.1.1.7",
    #         msg="SOPClassUID does not match Secondary Capture Image Storage",
    #     )
    
    # @pytest.mark.req27
    # @pytest.mark.ver13
    # @pytest.mark.post_build
    # @pytest.mark.inference
    # def test_do_not_match_pet(self):
    #     """
    #     REQ-27: SubtleGAD shall confirm input images are denoted as MR images in the file's metadata.
    #     """
    #     # run test
    #     output_folder = self.pre_test("REQ27")

    #     # change data modality
    #     input_folder_pt = os.path.join(self.tmp_folder, "input_pt")
    #     os.makedirs(input_folder_pt, exist_ok=True)
        
    #     self.change_modality(self.input_folder_zero, input_folder_pt)


    #     completed_process = self.run_inference(
    #         input_folder_pt,
    #         output_folder,
    #         self.config_file_small,
    #         self.license_file,
    #     )

    #     # check that execution passed
    #     self.validate_secondary_capture_generic(
    #         11, output_folder, completed_process
    #     )

    #     # remove output folder
    #     shutil.rmtree(output_folder)
    #     shutil.rmtree(input_folder_pt)

    # @pytest.mark.post_build
    # @pytest.mark.inference
    # def test_reject_misconfig(self):
    #     """
    #     Internal test for invalid modality inputs.
    #     """
    #     # run test
    #     output_folder = self.pre_test("REQ_MISCONFIG")

    #     # de-identify data
    #     input_folder_misconfig = os.path.join(
    #         self.tmp_folder, "input_misconfig"
    #     )
    #     os.makedirs(input_folder_misconfig, exist_ok=True)
    #     misconfig_small = self.change_modality_config(
    #         self.config_file_small, input_folder_misconfig
    #     )

    #     completed_process = self.run_inference(
    #         self.input_folder,
    #         output_folder,
    #         misconfig_small,
    #         self.license_file,
    #     )

    #     # check that execution passed
    #     self.validate_secondary_capture_generic(
    #         44, output_folder, completed_process
    #     )

    #     # remove output folder
    #     shutil.rmtree(output_folder)
    #     shutil.rmtree(input_folder_misconfig)


    # @pytest.mark.post_build
    # @pytest.mark.inference
    # @pytest.mark.prefix
    # def test_prefix_check(self):
    #     """
    #     SubtleGAD shall upgrade its uid-prefix based on the user's requirement suggested in the config. 
    #     """
    #     output_folder = self.pre_test('REQ111')

    #     with open(self.config_file, 'r') as file:
    #         config_keys =yaml.safe_load(file)

    #     ##Adding a new custom uid prefix 
    #     config_keys.update(uid_prefix = str("1.2.3.4.5.6.7."))
        
    #     ##Writing the updated config to a new path
    #     with open(self.config_copy, 'w') as file:
    #         yaml.dump(config_keys, file)

    #     ##Running SubtleGAD with the new config
    #     completed_process = self.run_inference(
    #         self.input_folder,
    #         output_folder,
    #         self.config_copy,
    #         self.license_file,
    #     )

    #     #Running the test successfully
    #     self.assertEqual(
    #         completed_process.returncode,
    #         0,
    #         msg="Execution failed: {}".format(completed_process.args),
    #     )
        

    #     output_series_dict = dicomscan(output_folder)
    #     output_series = list(output_series_dict.values())[0].get_list_sorted_datasets()[0][0]
    #     prefix_length = len(config_keys['uid_prefix'])

    #     ##Compare the new uid prefix with the saved output dicoms
    #     if 'uid_prefix' in config_keys:
    #         self.assertEqual(output_series.SeriesInstanceUID[:prefix_length], config_keys['uid_prefix'], msg= 'UID Prefix Dicom Tag was not updated in the output')

    #     shutil.rmtree(output_folder)


    # @pytest.mark.req15
    # @pytest.mark.ver3
    # @pytest.mark.post_build
    # @pytest.mark.inference
    # @pytest.mark.size_difference
    # def test_size_difference(self):
    #     """
    #     REQ-15: SubtleGAD should smoothly work on zd & ld Images of different size.
    #     """
    #     output_folder = self.pre_test("REQ15")
    #     completed_process = self.run_inference(
    #         self.input_zd_dicom,
    #         output_folder,
    #         self.config_file,
    #         self.license_file,
    #     )

    #     #Test the process with zd > ld in dimensions
    #     self.assertEqual(
    #         completed_process.returncode,
    #         0,
    #         msg="Execution failed: {}".format(completed_process.args),
    #     )

    #     # remove output folder
    #     shutil.rmtree(output_folder)

    #     #Test the process with zd > ld in dimensions
    #     output_folder = self.pre_test("REQ15")
    #     completed_process = self.run_inference(
    #         self.input_ld_dicom,
    #         output_folder,
    #         self.config_file,
    #         self.license_file,
    #     )

    #     self.assertEqual(
    #         completed_process.returncode,
    #         0,
    #         msg="Execution failed: {}".format(completed_process.args),
    #     )

    #     # remove output folder
    #     shutil.rmtree(output_folder)

    


    
if __name__ == "__main__":
    unittest.main()