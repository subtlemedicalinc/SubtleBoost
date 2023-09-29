#!/usr/bin/env python3
"""Script interface of the inference pipeline

@author: Srivathsa Pasumarthi <srivathsa@subtlemedical.com>
Copyright (c) Subtle Medical, Inc. (https://subtlemedical.com)
Created on 2020/02/06
"""

import os
from typing import Tuple, Optional
import hashlib
import traceback

from subtle.dcmutil.dicom_filter import DicomFilter
from subtle.util.subtle_app import SubtleApp
import subtle_gad_jobs
import tensorflow.compat.v1 as tf
import pdb
import json
import torch
torch.manual_seed(0)

tf.disable_v2_behavior()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print('SCRIPT DIT', SCRIPT_DIR)

class SubtleGADApp(SubtleApp):
    """The SubtleGAD App class"""

    def __init__(self):
        manifest_info = os.path.join(SCRIPT_DIR, "manifest.json")
        super().__init__(manifest_info, app_desc="The SubtleGAD App")

    def _validate_config(self) -> Tuple[int, str]:
        """
        Validate the config dict

        :return: an exit code and exit detail
        """
        self._logger.info('App config: %s', self._config)
        exit_code, msg = super()._validate_config()
        if exit_code != 0:
            return exit_code, msg
        #
        try:
            for series_definition in self._config["series"]:
                if series_definition["modality"].lower() != "mr":
                    return (
                        44,
                        "Series modality must be MR ({} found in {})".format(
                            series_definition["modality"],
                            series_definition["name"],
                        ),
                    )
                # TODO: validate compatible regex match ?

        # here we want to catch any potential error and return it as an unknown configuration error
        # pylint: disable=broad-except
        except Exception as exc:
            self._logger.error(
                "An error occured validating config: %s", exc, exc_info=True
            )
            return 40, self.TECHNICAL_ERROR_MSG

        return 0, ""

    def run(
        self,
        input_path: str,
        output_path: str,
        config_file: Optional[str] = "./config.yml",
        license_file: Optional[str] = None,
    ):
        """
        Override the SubtleApp.run() method

        :param input_path: the path to the input file
        :param output_path: the path to the output file
        :param config_file: optional path to the configuration file
        :param license_file: optional path to the license file
        :return: the exit code and the exit detail
        """
        
        exit_info = []

        # update output path
        output_path_dicom = os.path.join(output_path, self._out_dicom_dir)
        # create list of tasks based on dicom input
        dicom_filter_obj = DicomFilter(self._config, methodname= "itk")
        tasks, unmatched_series = dicom_filter_obj.process_incoming(input_path)

        # check valid number of tasks are found
        if not tasks:
            return [(11, "No job matched in the input data", None)]

        if (
            unmatched_series and self._config["enable_multiple_jobs"] and
            self._config["warn_unmatched_series"]
        ):
            # if some series were not matched with any job, notify the user with a warning:
            for series in unmatched_series:
                exit_detail = "Series did not match any job definition: {} ({} slices - \
                UID:{})".format(
                    series.seriesdescription,
                    series.nslices,
                    series.seriesinstanceUID,
                )
                exit_info.append((13, exit_detail, list(series.datasets)[0][0]))
        elif len(tasks) > 1 and not self._config["enable_multiple_jobs"]:
            exit_detail = "Multiple jobs matched in the input data and 'enable_multiple_jobs' was\
            set to {}".format(self._config["enable_multiple_jobs"])
            exit_info.append((12, exit_detail, None))

        # execute each task

        for i_task, task in enumerate(tasks):
            task_id = "{}:{}".format(i_task, task.job_name)
            try:
                # update execution specific configuration with app-wide configuration
                task.job_definition.exec_config.update(self._config)
                # clear exec_config from unused args
                _ = task.job_definition.exec_config.pop('jobs')
                _ = task.job_definition.exec_config.pop('series')

                # save model version to file as an info to platform
                model_id = task.job_definition.exec_config["model_id"]
                self._save_model_ver(model_id)

                model_dir = os.path.join(SCRIPT_DIR, "models", model_id)
                #print('directory of model ', model_dir)
                if not os.path.isdir(model_dir):
                    raise FileNotFoundError("Model Directory {} does not exist".format(model_dir))

                #pdb.set_trace()
                # create the task's job object and execute it
                job_obj = subtle_gad_jobs.SubtleGADJobType(
                    task=task,
                    model_dir=model_dir,
                    decrypt_key_hex=hashlib.sha256(
                        "{appName}::{appId}::{version}".format(
                            appName=self._app_name,
                            appId=self._app_id,
                            version=self._app_version,
                        ).encode()
                    ).hexdigest(),
                )

                job_obj(output_path_dicom)
                exit_info.append((0, "Task {} finished with success!".format(task_id), None))

            # pylint: disable=broad-except
            except Exception:
                exit_info.append((
                    100, "Task {} had an unexpected App error: {}".format(
                        task_id, traceback.format_exc()
                    ), list(task.list_series[0].datasets)[0][0]
                ))
                # continue to next task in the loop
                continue

        return exit_info


if __name__ == "__main__":
    import sys

    
    APP = SubtleGADApp()
    EXIT_CODE = APP.start()

    sys.exit(EXIT_CODE)

    
