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
        self.test_id = "SubtleBoost-" + os.environ.get("POST_TEST_TAG", "UNKNOWN")
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
        self.list_input = [os.path.join(self.path_data, "Boost_test")]
        self.input_folder = os.path.join(self.path_data, "Boost_test")

        #Zero dose and low dose dataset
        self.input_folder_zero = os.path.join(self.path_data, "Boost_test/OSag_3D_T1BRAVO_zero_dose_7/")
        self.input_folder_low = os.path.join(self.path_data, "Boost_test/OSag_3D_T1BRAVO_low_dose_8/")

        #PHI Free Dataset
        self.input_folder_no_PHI = os.path.join(self.path_data, "Boost_test_PHI")

        #Register Failure Dataset
        self.input_register_failure_folder = os.path.join(self.path_data, "register_failure")

        #Non match data
        self.input_non_match = os.path.join(self.path_data, "Boost_non_match")

        #Metadata incompatible data
        self.input_meta_data = os.path.join(self.path_data, "Boost_metadata")

        #Large input folder
        self.input_folder_large = os.path.join(self.path_data, "Boost_large_data")
        self.input_large_low = os.path.join(self.path_data, "Boost_large_data/10_AX_BRAVO_+C_Pre_Load_10_ld")

        #Config files
        self.config_file = os.path.join(self.build_dir, "config.yml")

        #Output Folder
        self.tmp_folder = tempfile.mkdtemp(dir=self.path_data)
        self.output_folder = os.path.join(self.tmp_folder, "output")

        global output_f
        output_f = self.output_folder

        #License File
        license_info = generate_license(
            12000,
            "SubtleBOOST",
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
    
    @pytest.mark.req33
    @pytest.mark.ver17
    @pytest.mark.post_build
    @pytest.mark.inference
    def test_no_gpu_available(self):
        """
        REQ-33: SubtleBoost confirm GPU and software compatibility prior to executing a job.
        """
        output_folder = self.pre_test("REQ33")

        #run_test
        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_file_small,
            self.license_file,
            "-1"
        )

        # check that execution passed
        self.assertEqual(
            completed_process.returncode,
            33,
            msg="Execution didn't fail as expected: {}".format(completed_process.args),
        )

        # remove output folder
        shutil.rmtree(output_folder)
    
    @pytest.mark.post_build
    @pytest.mark.inference
    def test_output_different_directory(self):
        """
        REQ-33: SubtleBoost to show input and output DICOMs reside in different directories.
        """
        output_folder = self.pre_test("REQ15")

        #run_test
        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_file_small,
            self.license_file,
        )

        # check that execution passed
        self.assertNotEqual(
            self.input_folder,
            output_folder,
            msg="Execution input and output directory are the same: {}".format(completed_process.args),
        )

        data_out = self.get_array_from_dir(output_folder)
        data_exp = self.get_array_from_dir(self.input_folder_low)

        self.assertEqual(data_out.shape, data_exp.shape,
                         "Processed data does not have the expected shape")

        # remove output folder
        shutil.rmtree(output_folder)

    @pytest.mark.post_build
    @pytest.mark.inference
    @pytest.mark.large
    def test_large_case_inference(self):
        """
        REQ-10: SubtleBoost to process a large case smoothly
        """
        output_folder = self.pre_test("REQ15")

        #run_test
        completed_process = self.run_inference(
            self.input_folder_large,
            output_folder,
            self.config_file_small,
            self.license_file,
        )

        # check that execution passed
        self.assertEqual(
            completed_process.returncode,
            0,
            msg="Execution didn't process as expected: {}".format(completed_process.args),
        )

        data_out = self.get_array_from_dir(output_folder)
        data_exp = self.get_array_from_dir(self.input_large_low)

        self.assertEqual(data_out.shape, data_exp.shape,
                         "Processed data does not have the expected shape")

        # remove output folder
        shutil.rmtree(output_folder)
    
    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_dicom_in(self):
        """
        test the infer script of SubtleBoost completes execution
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

    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_non_match(self):
        """
            Test to ensure that sequences with incorrect SeriesDescription do not match.
        """
        output_folder = self.pre_test("REQ3")

        #Run SubtleSynth with a failure mode case
        completed_process = self.run_inference(
            self.input_non_match,
            output_folder,
            self.config_file,
            self.license_file,
        )
        self.assertNotEqual(0, completed_process.returncode , msg= 'Series non matching was not captured')


    
    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_metadatacompatibility(self):
        """
            Test to ensure that metadata tags are corrrectly compared. 
        """

        with open(self.config_file, 'r') as file:
            config_keys =yaml.safe_load(file)
        ##Test MagneticFieldStrength
        output_folder = self.pre_test("REQ31")

        edit_config = copy.deepcopy(config_keys)

        edit_config['jobs'][0].update(magnetic_fieldstrength_tolerance = True)

        with open('./configmeta.yml', 'w') as file:
            yaml.dump(edit_config, file)
        
        #Run SubtleBoost with a failure mode case
        completed_process = self.run_inference(
            self.input_meta_data,
            output_folder,
            os.path.join(self.build_dir, "configmeta.yml"),
            self.license_file,
        )
        self.assertNotEqual(0, completed_process.returncode , msg= 'MagneticFieldStrength compatibility failure was not captured')

        # remove output folder
        shutil.rmtree(output_folder)

        ##Test the Manufacturer Model Name
        output_folder = self.pre_test("REQ31")

        edit_config = copy.deepcopy(config_keys)

        edit_config['jobs'][0].update(manufacturermodelname_flag = False)

        with open('./configmeta.yml', 'w') as file:
            yaml.dump(edit_config, file)
        
        #Run SubtleBoost with a failure mode case
        completed_process = self.run_inference(
            self.input_meta_data,
            output_folder,
            os.path.join(self.build_dir, "configmeta.yml"),
            self.license_file,
        )
        self.assertNotEqual(0, completed_process.returncode , msg= 'ManufacturerModelName compatibility failure was not captured')

        # remove output folder
        shutil.rmtree(output_folder)

        ##Test the Protocol Name
        output_folder = self.pre_test("REQ31")

        edit_config = copy.deepcopy(config_keys)

        edit_config['jobs'][0].update(protocolname_flag = False)

        with open('./configmeta.yml', 'w') as file:
            yaml.dump(edit_config, file)
        
        #Run SubtleBoost with a failure mode case
        completed_process = self.run_inference(
            self.input_meta_data,
            output_folder,
            os.path.join(self.build_dir, "configmeta.yml"),
            self.license_file,
        )
        self.assertNotEqual(0, completed_process.returncode , msg= 'ProtocolName compatibility failure was not captured')

        # remove output folder
        shutil.rmtree(output_folder)

        ##Test the Slice Thickness
        output_folder = self.pre_test("REQ31")

        edit_config = copy.deepcopy(config_keys)

        edit_config['jobs'][0].update(slice_thickness_tolerance = 0.01)

        with open('./configmeta.yml', 'w') as file:
            yaml.dump(edit_config, file)
        
        #Run SubtleBoost with a failure mode case
        completed_process = self.run_inference(
            self.input_meta_data,
            output_folder,
            os.path.join(self.build_dir, "configmeta.yml"),
            self.license_file,
        )
        self.assertNotEqual(0, completed_process.returncode , msg= 'SliceThickness compatibility failure was not captured')

        # remove output folder
        shutil.rmtree(output_folder)

        ##Test the Pixel Spacing
        output_folder = self.pre_test("REQ31")

        edit_config = copy.deepcopy(config_keys)

        edit_config['jobs'][0].update(pixelspacing_tol = 0.01)

        with open('./configmeta.yml', 'w') as file:
            yaml.dump(edit_config, file)
        
        #Run SubtleBoost with a failure mode case
        completed_process = self.run_inference(
            self.input_meta_data,
            output_folder,
            os.path.join(self.build_dir, "configmeta.yml"),
            self.license_file,
        )
        self.assertNotEqual(0, completed_process.returncode , msg= 'PixelSpacing compatibility failure was not captured')

        # remove output folder
        shutil.rmtree(output_folder)

        ##Test the Field of View
        output_folder = self.pre_test("REQ31")

        edit_config = copy.deepcopy(config_keys)

        edit_config['jobs'][0].update(fov_tolerance = 0.01)

        with open('./configmeta.yml', 'w') as file:
            yaml.dump(edit_config, file)
        
        #Run SubtleBoost with a failure mode case
        completed_process = self.run_inference(
            self.input_meta_data,
            output_folder,
            os.path.join(self.build_dir, "configmeta.yml"),
            self.license_file,
        )
        self.assertNotEqual(0, completed_process.returncode , msg= 'FieldofView compatibility failure was not captured')

        # remove output folder
        shutil.rmtree(output_folder)
        
    @pytest.mark.req30
    @pytest.mark.ver8
    @pytest.mark.post_build
    @pytest.mark.inference
    def test_metadata(self):
        """
        REQ-30: SubtleBoost shall not alter Boost image metadata that is not required to 1) produce a new series or
        2) indicate that the series has been enhanced via SubtleBoost
        List the excluded dicom tags
        """
        output_folder = self.pre_test("REQ30")
        
        #run_test
        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_file_small,
            self.license_file,
        )

        # check that execution passed
        self.assertEqual(
            completed_process.returncode,
            0,
            msg="Execution failed: {}".format(completed_process.args),
        )


        #checking a sample output dicom metadata
        output_series_dict = dicomscan(output_folder)
        output_series = list(output_series_dict.values())[0].get_list_sorted_datasets()[0]

        input_series_dict = dicomscan(self.input_folder_low)
        input_series = list(input_series_dict.values())[0].get_list_sorted_datasets()[0]
        
        #Excluding the following metadata:
        #SOP Instance UID
        #Study Description
        #Image Type
        #Series Description
        #Acquisition Matrix
        #Series Number 
        #Series InstanceUID
        #Protocol Name
        #Rows Columns
        
        exclude_list = {0x8: [0x18, 0x103e, 0x1030 ,0x8],
                        0x18:[0x1310, 0x1030],

                        0x20: [0xe, 0x11, 0x106],
                        0x7fe0: [0x10],
                        0x28: [0x107, 0x106, 0x10, 0x11],
                        }

        
        for i in range(len(input_series)):
            input_dcm = input_series[i]
            output_dcm = output_series[i]

            for elem in input_dcm:
                group = hex(elem.tag.group)
                elem_id = hex(elem.tag.elem)

                if int(group, 16) in exclude_list.keys() and int(elem_id, 16) in exclude_list[int(group, 16)]:
                    pass
                else:
                    input_val = input_dcm[group, elem_id]
                    try:  
                        output_val = output_dcm[group, elem_id]
                    except KeyError:
                        continue
                    #Comparing the Input T2 dicom metadata equal to Output Dicom metadata
                    assert(input_val == output_val )

        # remove output folder
        shutil.rmtree(output_folder)
    
    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_output_different_series_uid(self):
        """
        REQ-20: SubtleApp shall update series instance for output image to ‘Processed by SubtleApp’ and
        assign a unique new SeriesInstanceUID for output image.
        """
        MainPostBuild.t_output_different_series_uid(self)
    
    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_suffix(self):
        """
        REQ-20: SubtleApp shall update series instance for output image to ‘Processed by SubtleApp’ and
        assign a unique new SeriesDescription for output image.
        """
        with open(self.config_file, 'r') as file:
            config_keys =yaml.safe_load(file)

        output_folder = self.pre_test("REQ20")

        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_file,
            self.license_file,
        )
        # check that execution passed
        self.assertEqual(
            completed_process.returncode,
            0,
            msg="Execution failed: {}".format(completed_process.args),
        )
        # check that SeriesDescription output includes the suffix

        output_walk = list(os.walk(os.path.join(self.build_dir, output_folder)))
        instance_out = os.path.join(output_walk[-1][0], output_walk[-1][-1][0])

        series_desc_out = pydicom.read_file(
            instance_out, stop_before_pixels=True
        ).SeriesDescription
        series_desc_suffix = config_keys['jobs'][0]['exec_config']['series_desc_suffix']
        self.assertTrue(
                series_desc_out.endswith(series_desc_suffix),
                msg="Identical SeriesDescription between input and output series",
        )
        data_out = self.get_array_from_dir(output_folder)
        data_exp = self.get_array_from_dir(self.input_folder_low)

        self.assertEqual(data_out.shape, data_exp.shape,
                         "Processed data does not have the expected shape")

        # remove output folder
        shutil.rmtree(output_folder)

    @pytest.mark.post_build
    @pytest.mark.subtleapp
    def test_output_different_series_description(self):
        """
        REQ-20: SubtleApp shall update series instance for output image to ‘Processed by SubtleApp’ and
        assign a unique new SeriesDescription for output image.
        """
        MainPostBuild.t_output_different_series_description(self)

    
    @pytest.mark.req38
    @pytest.mark.ver27
    @pytest.mark.post_build
    @pytest.mark.enhanced_dicom
    def test_enhanced_dicom(self):
        """
        REQ-38 : SubtleBoost shall operate on enhanced DICOMs. 
        """

        #run test
        output_folder = self.pre_test("REQ38")
        completed_process = self.run_inference(
            self.input_enhanced_dicom,
            output_folder,
            self.config_file_small,
            self.license_file,
        )
        # check that execution passed
        self.assertEqual(
            completed_process.returncode,
            0,
            msg="Execution failed: {}".format(completed_process.args),
        )
        # get data
        data_out = self.get_array_from_dir(output_folder)
        data_exp = self.get_array_from_dir(self.expected_enhanced_dicom)

        self.assertEqual(data_out.shape, data_exp.shape,
                         "Processed data does not have the expected shape")

        self.assertTrue(np.allclose(data_out, data_exp, atol=10000, rtol=10000),
                        "Processed data is different than expected data")
        # remove output folder
        shutil.rmtree(output_folder)

    @pytest.mark.req36
    @pytest.mark.ver25
    @pytest.mark.post_build
    @pytest.mark.inference
    @pytest.mark.minimal_test
    def test_compressed_dicom(self):
        """
        REQ-36: SubtleBoost shall operate on compression based dicoms. Test that compressed dicom studies are read, processed, and output saved
        """    

        return MainPostBuild.t_compressed_dicom_processing(self)
    
    @pytest.mark.req37
    @pytest.mark.ver26
    @pytest.mark.post_build
    @pytest.mark.inference
    @pytest.mark.register_fail
    def test_register_failure(self):
        """
        REQ-37: SubtleBoost shall provide an error message when registration fails. 
        """
        return MainPostBuild.t_register_failure(self)

    @pytest.mark.req34
    @pytest.mark.ver16
    @pytest.mark.post_build
    @pytest.mark.inference
    def test_reject_license(self):
        """
        REQ-6: SubtleBoost shall have CLI detailed the attachment
        """
        # run test
        output_folder = self.pre_test("REQ16")

        # create license
        input_folder_license = os.path.join(self.tmp_folder, "input_license")
        os.makedirs(input_folder_license, exist_ok=True)

        # empty
        license_info = {"SerialNumber": "", "Expiration": "", "LicenseKey": ""}
        license_file = os.path.join(input_folder_license, "license_empty.json")
        # write license
        with open(license_file, "w") as f:
            json.dump(license_info, f)
        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_file_small,
            license_file,
        )
        # check that execution failed correctly
        self.validate_secondary_capture_generic(
            90, output_folder, completed_process
        )

        
        # wrong key
        license_info = generate_license(
            6000,
            "SubtleBOOST",
            self.SERIAL_NUMBER,
            date.today() + timedelta(days=2),
        )
        license_info["LicenseKey"] = "xxx"
        license_file = os.path.join(
            input_folder_license, "license_wrong_key.json"
        )
        # write license
        with open(license_file, "w") as f:
            json.dump(license_info, f)
        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_file_small,
            license_file,
        )
        # check that execution failed correctly
        self.validate_secondary_capture_generic(
            90, output_folder, completed_process
        )

        
        # expired
        license_info = generate_license(
            12000,
            "SubtleBOOST",
            self.SERIAL_NUMBER,
            date.today() - timedelta(days=2),
        )
        license_file = os.path.join(
            input_folder_license, "license_expired.json"
        )
        # write license
        with open(license_file, "w") as f:
            json.dump(license_info, f)
        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_file_small,
            license_file,
        )
        # check that execution failed correctly
        self.validate_secondary_capture_generic(
            90, output_folder, completed_process
        )

        # remove output folder
        shutil.rmtree(output_folder)
        shutil.rmtree(input_folder_license)


    @pytest.mark.req27
    @pytest.mark.ver13
    @pytest.mark.post_build
    @pytest.mark.inference
    def test_do_not_match_pet(self):
        """
        REQ-27: SubtleBoost shall confirm input images are denoted as MR images in the file's metadata.
        """
        # run test
        output_folder = self.pre_test("REQ27")

        # change data modality
        input_folder_pt = os.path.join(self.tmp_folder, "input_pt")
        os.makedirs(input_folder_pt, exist_ok=True)
        
        self.change_modality(self.input_folder_zero, input_folder_pt)


        completed_process = self.run_inference(
            input_folder_pt,
            output_folder,
            self.config_file_small,
            self.license_file,
        )

        # check that execution passed
        self.validate_secondary_capture_generic(
            11, output_folder, completed_process
        )

        # remove output folder
        shutil.rmtree(output_folder)
        shutil.rmtree(input_folder_pt)

    @pytest.mark.post_build
    @pytest.mark.inference
    def test_reject_misconfig(self):
        """
        Internal test for invalid modality inputs.
        """
        # run test
        output_folder = self.pre_test("REQ_MISCONFIG")

        # de-identify data
        input_folder_misconfig = os.path.join(
            self.tmp_folder, "input_misconfig"
        )
        os.makedirs(input_folder_misconfig, exist_ok=True)
        misconfig_small = self.change_modality_config(
            self.config_file_small, input_folder_misconfig
        )

        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            misconfig_small,
            self.license_file,
        )

        # check that execution passed
        self.validate_secondary_capture_generic(
            44, output_folder, completed_process
        )

        # remove output folder
        shutil.rmtree(output_folder)
        shutil.rmtree(input_folder_misconfig)


    @pytest.mark.post_build
    @pytest.mark.inference
    @pytest.mark.prefix
    def test_prefix_check(self):
        """
        SubtleBoost shall upgrade its uid-prefix based on the user's requirement suggested in the config. 
        """
        output_folder = self.pre_test('REQ111')

        with open(self.config_file, 'r') as file:
            config_keys =yaml.safe_load(file)

        ##Adding a new custom uid prefix 
        config_keys.update(uid_prefix = str("1.2.3.4.5.6.7."))
        
        ##Writing the updated config to a new path
        with open(self.config_copy, 'w') as file:
            yaml.dump(config_keys, file)

        ##Running SubtleBoost with the new config
        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_copy,
            self.license_file,
        )

        #Running the test successfully
        self.assertEqual(
            completed_process.returncode,
            0,
            msg="Execution failed: {}".format(completed_process.args),
        )
        

        output_series_dict = dicomscan(output_folder)
        output_series = list(output_series_dict.values())[0].get_list_sorted_datasets()[0][0]
        prefix_length = len(config_keys['uid_prefix'])

        ##Compare the new uid prefix with the saved output dicoms
        if 'uid_prefix' in config_keys:
            self.assertEqual(output_series.SeriesInstanceUID[:prefix_length], config_keys['uid_prefix'], msg= 'UID Prefix Dicom Tag was not updated in the output')

        shutil.rmtree(output_folder)
    
    @pytest.mark.post_build
    @pytest.mark.inference
    def test_processing_pipeline(self):
        """
        REQ-39: SubtleBoost shall have a configurable parameter to permit no registration.
        """
        output_folder = self.pre_test("REQ39")

        
        #Add custom config parameters to the config in the preprocessing pipeline and the post processing pipeline
        with open(self.config_file_small, 'r') as file:
            config_keys =yaml.safe_load(file)
        
        config_keys['jobs'][0]['exec_config'].update(pipeline_preproc = ({'STEP1' : {'op' : 'MASK'} , 'STEP2' : {'op' : 'SKULLSTRIP'}, 'STEP3' : {'op' : 'REGISTER'},'STEP4' : {'op' : 'HIST'}, 'STEP5' : {'op' : 'SCALETAG'}, 'STEP6' : {'op' : 'SCALEGLOBAL'}, 'STEP7' : {'op' : 'CLIP'} }))
        # config_keys['jobs'][0]['exec_config'].update(pipeline_preproc = ({}))
        # config_keys['jobs'][0]['exec_config'].update(pipeline_preproc = ({}))
        # config_keys['jobs'][0]['exec_config'].update(pipeline_preproc = ({}))
        # config_keys['jobs'][0]['exec_config'].update(pipeline_preproc = ({}))
        # config_keys['jobs'][0]['exec_config'].update(pipeline_preproc = ({}))
        # config_keys['jobs'][0]['exec_config'].update(pipeline_preproc = ({}))

        config_keys['jobs'][0]['exec_config'].update(pipeline_postproc = ({'STEP1' : {'op' : 'RESCALEGLOBAL'} ,'STEP2' : {'op' : 'RESCALEDICOM'}}))
        #config_keys['jobs'][0]['exec_config'].update(pipeline_postproc = ({}))
        
        
        print(config_keys)
        with open(self.config_copy, 'w') as file:
            yaml.dump(config_keys, file)

        # run_test
        completed_process = self.run_inference(
            self.input_folder,
            output_folder,
            self.config_copy,
            self.license_file
        )
        # check that execution passed
        self.assertEqual(
            completed_process.returncode,
            0,
            msg="Execution didn't succeed as expected: {}".format(completed_process.args),
        )

        ##check the output shape
        # get data
        data_out = self.get_array_from_dir(output_folder)
        data_exp = self.get_array_from_dir(self.input_folder_low)

        self.assertEqual(data_out.shape, data_exp.shape,
                         "Processed data does not have the expected shape")

        ## output data is different from T2
        self.assertFalse(np.allclose(data_out, data_exp, atol=0.1),
                        "Processed data is different than expected data")

        # remove output folder
        shutil.rmtree(output_folder)

    
if __name__ == "__main__":
    unittest.main()