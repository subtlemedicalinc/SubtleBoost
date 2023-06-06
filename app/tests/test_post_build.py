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
from infer import SubtleGADApp
from subtle.util.data_loader import dicomscan
from subtle.util.licensing import generate_license
import pdb
from collections import OrderedDict


class PostBuildTest(unittest.TestCase):
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
    
    def setUp(self):
        # model files
        self.test_id = "SubtleSynth-" + os.environ.get("POST_TEST_TAG", "UNKNOWN")
        this_dir = os.path.dirname(__file__)
        
        self.build_dir = os.path.abspath(os.path.join(this_dir, "../../dist"))

        path_data = os.path.join(
            os.path.abspath(this_dir), "post_build_test_data"
        )

        # input and output data
        # list of tuple (data_folder, roi_csv)
        self.list_input = [os.path.join(path_data, "Gad_test")]
        self.input_folder = os.path.join(path_data, "Gad_test")

        #Config files
        self.config_file = os.path.join(self.build_dir, "config.yml")

        print('what is config file ', self.config_file)
        #Output Folder
        self.tmp_folder = tempfile.mkdtemp(dir=path_data)
        self.output_folder = os.path.join(self.tmp_folder, "output")

        #License File
        license_info = generate_license(
            4000,
            "SubtleGAD",
            self.SERIAL_NUMBER,
            date.today() + timedelta(days=2),
        )

        self.license_file = os.path.join(path_data, "license.json")
        # write license
        with open(self.license_file, "w") as f:
            json.dump(license_info, f)
        
    def pre_test(self, test_name):
        
        output_folder = os.path.abspath(
            os.path.join(self.output_folder, test_name)
        )
        os.makedirs(output_folder, exist_ok=True)
        os.chmod(output_folder, 777)
        return output_folder

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
            "python3.10 /home/SubtleGad/app/infer.py {input_folder} {output_folder} "
            "-c {config} -l {license}"
        )

        cmd = cmd_str.format(
            cuda=cuda_device,
            #interface=os.path.join(self.build_dir, interface_script),
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

    # @pytest.mark.minimal_test
    # @pytest.mark.post_build
    # def test_executable_file(self):
        
    #     completed_process = subprocess.run(
    #         "./infer/infer --help", cwd=self.build_dir, shell=True
    #     )
    #     self.assertEqual(
    #         completed_process.returncode,
    #         0,
    #         msg="Execution failed: {}".format(completed_process.args),
    #     )

    
    @pytest.mark.post_build
    @pytest.mark.inference
    def test_dicom_in(self):
        """
        UN-5 The Technologist and Radiologist users need SubtleSynth to
        operate with Digital Imaging and Communications in Medicine (DICOM) version 3 image inputs and outputs
        """
        
        # run test
        output_folder = self.pre_test("UN5")
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

        # remove output folder
        shutil.rmtree(output_folder)
    

if __name__ == "__main__":
    unittest.main()