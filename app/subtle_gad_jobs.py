"""The SubtleGAD jobs

@author: Srivathsa Pasumarthi <srivathsa@subtlemedical.com>
Copyright Subtle Medical, Inc. (https://subtlemedical.com)
Created on 2020/02/06
"""

import os
from typing import Dict, Optional
from collections import OrderedDict, namedtuple
from itertools import groupby

import pydicom
import pydicom.errors
import numpy as np

from subtle.util.inference_job_utils import BaseJobType, GenericInferenceModel
from subtle.procutil import preprocess_single_series, postprocess_single_series
from subtle.dcmutil import pydicom_utils, series_utils

class SubtleGADJobType(BaseJobType):
    """The job type of SubtleGAD dose reduction"""
    default_processing_config_gad = \
        {
            # app params
            "app_name": "SubtleGAD",
            "app_id": 3000,
            "app_version": "0.0.0",

            # model params
            "model_id": "None",

            # general params
            "not_for_clinical_use": True,

            # preprocessing params:
            "transform_type": "rigid",
            "normalize": True,
            "normalize_fun": "mean",
            "noise_mask_threshold": 0.1,
            "noise_mask_area": True,
            "scale_matching": True,
            "skip_hist_norm": True,
            "skull_strip": True,
            "skull_strip_all_ims": True,

            # inference params:
            "inference_mpr": True,
            "num_rotations": 5,
            "slices_per_input": 7,
            "resize": 512,

            # post processing params
            "series_desc_prefix": "",
            "series_desc_suffix": "",
            "series_number_offset": 100,
         }

    SubtleGADProcConfig = namedtuple("SubtleGADProcConfig", ' '.join(list(default_processing_config_gad.keys())))

    def __init__(self, task, model_dir, decrypt_key_hex: Optional[str] = None):
        """Initialize the job type object

        :param task: subtle.dcmutils.dicom_filter.Task object of the task to execute
        :param model_dir: directory of the model to load for this job
        :param decrypt_key_hex: the decrypt key
        """

        name = task.job_name
        config = task.job_definition.exec_config
        BaseJobType.__init__(self, name, config)

        self._logger.info("Loading model from %s...", model_dir)
        self._model = GenericInferenceModel(model_dir=model_dir,
                                            decrypt_key_hex=decrypt_key_hex)
        self._logger.info("Loaded %s model.", self._model.model_type)
        if self._model.model_type == 'invalid':
            raise NotImplementedError("Invalid model found in {}".format(model_dir))
        # update model config with app config:
        # to update tunable parameters, job parameters and app parameters
        self._model.update_config(self.config)

        self.task = task

        # define and prepare config
        proc_config = self.default_processing_config_gad.copy()

        proc_config.update(self._model.model_config)
        k = list(proc_config.keys())
        _ = [proc_config.pop(f) for f in k if f not in self.SubtleGADProcConfig._fields]

        self._proc_config = self.SubtleGADProcConfig(**proc_config)


        # initialize arguments to update during processing
        # dict of input datasets by frame
        # (keys = frame sequence name, values = list of pydicom datasets)
        self._input_datasets = {}  # reference dataset list

        # dict of input arrays by frame
        # (keys = frame sequence name, values = pixel array)
        self._raw_input = {}

        self._output_data = {}

    def _preprocess(self, _):
        """The preprocess func

        :param _: (in_dicom in BaseJobType by default)
        """
        self._get_dicom_data()
        #
        # get original pixel data and meta data info
        self._get_raw_pixel_data()

        # dictionary of the data to process by frame
        dict_pixel_data = self._preprocess_raw_pixel_data()

        return dict_pixel_data, {}

    # pylint: disable=arguments-differ
    def _process(self, dict_pixel_data, meta_data):
        """Process the pixel data with the meta data.

        :param dict_pixel_data: dictionary of the input pixel data (numpy arrays) by frame
        :param meta_data: a dict containing required meta data
        :return: the processed output data
        """
        dict_output_array = {}

        self._logger.info("starting inference (job type: %s)", self.name)

        for frame_seq_name, pixel_data in dict_pixel_data.items():
            # update pixel shape
            if pixel_data.ndim == 3:
                pixel_data = pixel_data[..., None]

            # update model input shape
            self._model.update_input_shape(pixel_data.shape)
            # perform inference with default input format (NHWC)
            output_array = self._model.predict(pixel_data)
            dict_output_array[frame_seq_name] = output_array

        self._logger.info("inference finished")

        return dict_output_array

    # pylint: disable=arguments-differ
    def _postprocess(self, dict_pixel_data, out_dicom_dir):
        """Postprocessing step of the job.

        :param dict_pixel_data: dict of the result pixel data in a numpy array by frame.
        :param out_dicom_dir: the directory or tarball of output DICOM files
        """
        # check input size
        for frame_seq_name in self._input_datasets[0]:

            # TEMP CODE
            dict_pixel_data[frame_seq_name] = np.repeat(dict_pixel_data[frame_seq_name], len(self._input_datasets[0][frame_seq_name]), axis=0)

            self._logger.info('postprocess shape %s', dict_pixel_data[frame_seq_name].shape)

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
        self._save_data(self._output_data, self._input_datasets[0], out_dicom_dir)

    def _get_dicom_data(self):
        """
        Get dicom datasets from the DicomSeries passed through the task as a dictionary of sorted
        datasets by frame: keys are frame sequence name and values are lists of pydicom datasets
        :return:
        """

        assert len(self.task.dict_required_series) == 2, "More than two input series found - only zero dose and low dose series are allowed"

        zero_dose_series = self.task.dict_required_series['zero_dose_series']
        low_dose_series = self.task.dict_required_series['low_dose_series']

        if not isinstance(zero_dose_series, series_utils.DicomSeries) or not \
        isinstance(low_dose_series, series_utils.DicomSeries):
            raise TypeError("Input series should be a DicomSeries object")
        # get dictionary of sorted datasets by frame
        self._input_datasets = (
            zero_dose_series.get_dict_sorted_datasets(),
            low_dose_series.get_dict_sorted_datasets()
        )

    def _get_raw_pixel_data(self):
        for frame_seq_name, _ in self._input_datasets[0].items():
            list_data_zero = [
                pydicom_utils.get_pixel_data_slice(dcm)[np.newaxis, :, :]
                for dcm in self._input_datasets[0][frame_seq_name]
            ]

            list_data_low = [
                pydicom_utils.get_pixel_data_slice(dcm)[np.newaxis, :, :]
                for dcm in self._input_datasets[1][frame_seq_name]
            ]

            zero_data_np = np.concatenate(list_data_zero, axis=0)
            low_data_np = np.concatenate(list_data_low, axis=0)

            self._raw_input[frame_seq_name] = np.array([zero_data_np, low_data_np])

            self._logger.info(
                "the shape of array {}".format(
                    self._raw_input[frame_seq_name].shape
                )
            )

    def _preprocess_raw_pixel_data(self):
        """
        get pixel data from dicom datasets
        and resample the pixel data to the specified shape
        :return: preprocessed pixel data
        """
        # apply preprocessing for each frame of the dictionary
        dict_input_data = {}

        for frame_seq_name, data_array in self._raw_input.items():
            input_data = data_array.copy()

            # write preprocess chain here
            zero_input = np.array([input_data[0, 180:187, ...].transpose(1, 2, 0)])
            low_input = np.array([input_data[1, 180:187, ...].transpose(1, 2, 0)])
            input_data = np.concatenate([zero_input, low_input], axis=3)

            print('input data shape', input_data.shape)

            dict_input_data[frame_seq_name] = input_data

        return dict_input_data

    def _postprocess_data(self, dict_pixel_data):
        """
        Post process the pixel data. i.e. rescale based on SUV scale
        and resample to reference size, pixel spacing and slice positions
        :param dict_pixel_data: dict of data to post process by frame
        :return: post processed data
        """

        for frame_seq_name, pixel_data in dict_pixel_data.items():
            # get 3D volume
            if len(pixel_data.shape) == 4:
                pixel_data = pixel_data[..., 0]

            # write post processing here
            self._output_data[frame_seq_name] = pixel_data

    def _save_data(self,
                   dict_pixel_data: Dict,
                   dict_template_ds: Dict,
                   out_dicom_dir: str,
                   enable_thru_plane: bool = False):
        """
        Save pixel data to the output directory
        based on a reference dicom dataset list
        :param dict_pixel_data: dict of array of data by frame to save to output directory
        :param dict_template_ds: dict of template dataset to use to save pixel data
        :param out_dicom_dir: path to output directory
        :return: None
        """
        # save data frame by frame but with the same series UID and a shared UID pool
        # create output directory
        out_dicom_dir = os.path.join(out_dicom_dir,
                                     list(dict_template_ds.values())[0][0].StudyInstanceUID,)
        os.makedirs(out_dicom_dir, exist_ok=True)  # do not complain if exists

        # generate a new series UID
        series_uid = pydicom_utils.generate_uid()
        uid_pool = set()
        uid_pool.add(series_uid)

        out_dicom_dir = os.path.join(out_dicom_dir, series_uid)
        os.makedirs(out_dicom_dir)

        i_instance = 1
        for iframe, frame_seq_name in enumerate(dict_template_ds.keys()):
            # remove unused dimension(s)
            pixel_data = np.squeeze(dict_pixel_data[frame_seq_name])

            # edit pixel data if not_for_clinical_use
            if self._proc_config.not_for_clinical_use:
                pixel_data = pydicom_utils.imprint_volume(pixel_data)

            # get slice position information
            nslices, nrow, ncol = pixel_data.shape

            # get slice location information if through-plane acceleration
            if enable_thru_plane:
                slice_thickness, list_slice_location, list_ipp = self._get_slice_location(
                    pixel_data, frame_seq_name
                )
            else:
                list_slice_location, list_ipp, slice_thickness = None, None, None

            # save all individual slices
            for i_slice in range(nslices):
                # get the right dataset reference
                if enable_thru_plane:
                    # get slice specific values
                    slice_loc = str(list_slice_location[i_slice])
                    ipp = list_ipp[i_slice]
                    instance_number = str(i_instance)

                    # get correct reference dataset
                    distance_to_slices = [
                        np.abs(ds.SliceLocation - slice_loc)
                        for ds in dict_template_ds[frame_seq_name]
                    ]
                    i_ref = distance_to_slices.index(min(distance_to_slices))
                    out_dataset = dict_template_ds[frame_seq_name][i_ref]

                    # set slice specific metadata
                    out_dataset.SliceLocation = slice_loc
                    out_dataset.ImagePositionPatient[0] = str(ipp[0])
                    out_dataset.ImagePositionPatient[1] = str(ipp[1])
                    out_dataset.ImagePositionPatient[2] = str(ipp[2])
                    out_dataset.InstanceNumber = instance_number
                    out_dataset.SliceThickness = slice_thickness

                else:
                    out_dataset = dict_template_ds[frame_seq_name][i_slice]

                # add model and app info to output dicom file
                for tag in self.private_tag:
                    out_dataset.add_new(*tag)

                # set series-wide metadata
                out_dataset.SeriesDescription = "{}{}{}".format(
                    self._proc_config.series_desc_prefix,
                    out_dataset.get("SeriesDescription", ""),
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

                # save new pixel data to dataset
                # new scale is always False for MR images
                slice_pixel_data = pixel_data[i_slice]
                pydicom_utils.save_float_to_dataset(
                    out_dataset, slice_pixel_data, new_rescale=False
                )

                # save in output folder
                out_path = os.path.join(
                    out_dicom_dir,
                    "IMG-{:04d}-{:04d}.dcm".format(iframe, i_slice),
                )
                out_dataset.save_as(out_path)
                i_instance += 1
