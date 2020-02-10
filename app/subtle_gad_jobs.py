"""The SubtleGAD jobs

@author: Srivathsa Pasumarthi <srivathsa@subtlemedical.com>
Copyright Subtle Medical, Inc. (https://subtlemedical.com)
Created on 2020/02/06
"""

import os
from typing import List, Union, Optional, get_type_hints
from collections import OrderedDict, namedtuple
from itertools import groupby

import pydicom
import pydicom.errors
import numpy as np

from subtle.util.inference_job_utils import BaseJobType, GenericInferenceModel
from subtle.procutil import preprocess_single_series, postprocess_single_series
from subtle.dcmutil import pydicom_utils


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
            "mask_threshold": 0.1,
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

    def __init__(self, name, config, model_dir, decrypt_key_hex: Optional[str] = None):
        """Initialize the job type object

        :param name: see BaseJobType
        :param config: see BaseJobType, contains the following key(s)
                       download_base_dir: The base directory
        :param model_dir: directory of the model to load for this job
        :param decrypt_key_hex: the decrypt key
        """
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

    def _preprocess(self, in_dicom):
        """The preprocess func

        :param in_dicom: see BaseJobType
        """
        self._get_dicom_data(in_dicom)
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
        for frame_seq_name in self._input_datasets:

            # TEMP CODE
            dict_pixel_data[frame_seq_name] = np.repeat(dict_pixel_data[frame_seq_name], len(self._input_datasets[frame_seq_name]), axis=0)

            self._logger.info('postprocess shape %s', dict_pixel_data[frame_seq_name].shape)

            if (
                    len(self._input_datasets[frame_seq_name])
                    != dict_pixel_data[frame_seq_name].shape[0]
            ):
                msg = "postprocess() got mismatched pixel data and template DICOM files"
                self._logger.error(msg)
                raise ValueError(msg)

        # data post processing
        dict_pixel_data_out = self._postprocess_data(dict_pixel_data)
        # data saving
        self._save_data(dict_pixel_data_out, out_dicom_dir)

    def _get_dicom_data(self, in_dicom: Union[str,
                                              List[List[str]],
                                              List[List[pydicom.dataset.FileDataset]]]):
        """
        read dicom data and create lists of dicom datasets
        :param in_dicom:
        :return:
        """
        in_dicom_files = []
        list_dcm_ds = []
        if isinstance(in_dicom, str):
            assert os.path.isdir(
                in_dicom
            ), "The input dicom str is not a directory"
            in_dicom_files = [
                os.path.join(in_dicom, f) for f in os.listdir(in_dicom)
            ]
        elif isinstance(in_dicom, list):
            assert len(in_dicom) == 1, "More than one input series"
            if isinstance(in_dicom[0][0], str):
                in_dicom_files = in_dicom[0]
            elif isinstance(in_dicom[0][0], pydicom.dataset.FileDataset):
                list_dcm_ds = in_dicom[0]

        if not in_dicom_files and not list_dcm_ds:
            raise TypeError(
                "Wrong input type. Supported types: {}".format(
                    get_type_hints(self._get_dicom_data)["in_dicom"]
                )
            )

        if not list_dcm_ds:
            for f in in_dicom_files:
                try:
                    list_dcm_ds.append(pydicom.read_file(f))
                except pydicom.errors.InvalidDicomError:
                    # ignore non dicom files
                    pass

        list_dcm_ds = sorted(list_dcm_ds, key=lambda ds: ds.InstanceNumber)

        def sorting_key(ds):
            return ds.get("SequenceName", "single_frame_sequence")

        list_all_dcm_ds = sorted(list_dcm_ds, key=sorting_key)
        dict_dcm_ds_by_frame = OrderedDict(
            [
                (seq_name, [ds for ds in grouper])
                for seq_name, grouper in groupby(list_all_dcm_ds, key=sorting_key)
            ]
        )

        # make sure all the frames have the same slices
        unique_slice_loc = set([ds.SliceLocation for ds in list_all_dcm_ds])
        for frame_seq_name, list_ds_frame in dict_dcm_ds_by_frame.items():
            assert (
                    set([ds.SliceLocation for ds in list_ds_frame])
                    == unique_slice_loc
            ), "Some slices are missing for frame {}".format(frame_seq_name)

        instance1_index = next(
            (
                index
                for (index, ds) in enumerate(list_all_dcm_ds)
                if str(ds.InstanceNumber) == "1"
            ),
            None,
        )
        instance2_index = next(
            (
                index
                for (index, ds) in enumerate(list_all_dcm_ds)
                if str(ds.InstanceNumber) == "2"
            ),
            None,
        )
        reverse_order = (
            list_all_dcm_ds[instance2_index].SliceLocation
            < list_all_dcm_ds[instance1_index].SliceLocation
            if instance1_index is not None and instance2_index is not None
            else False
        )

        # sort each dataset list by projected z pos
        dict_dcm_ds_by_frame = OrderedDict(
            [
                (
                    seq_name,
                    sorted(
                        list_ds_frame,
                        key=pydicom_utils.get_projected_z_pos,
                        reverse=reverse_order,
                    ),
                )
                for seq_name, list_ds_frame in dict_dcm_ds_by_frame.items()
            ]
        )

        self._input_datasets = dict_dcm_ds_by_frame

    def _get_raw_pixel_data(self):
        for frame_seq_name, list_ds_frame in self._input_datasets.items():
            list_data_fast = [
                pydicom_utils.get_pixel_data_slice(dcm)[np.newaxis, :, :]
                for dcm in list_ds_frame
            ]
            self._raw_input[frame_seq_name] = np.concatenate(
                list_data_fast, axis=0
            )
            #
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
            input_data = np.array([input_data[180:194, ...].transpose(1, 2, 0)])

            dict_input_data[frame_seq_name] = input_data

        return dict_input_data

    def _postprocess_data(self, dict_pixel_data):
        """
        Post process the pixel data. i.e. rescale based on SUV scale
        and resample to reference size, pixel spacing and slice positions
        :param dict_pixel_data: dict of data to post process by frame
        :return: post processed data
        """
        dict_pixel_data_out = {}
        for frame_seq_name, pixel_data in dict_pixel_data.items():
            # get 3D volume
            if len(pixel_data.shape) == 4:
                pixel_data = pixel_data[..., 0]

            # write post processing here

            dict_pixel_data_out[frame_seq_name] = pixel_data

        return dict_pixel_data_out


    def _save_data(
            self, dict_pixel_data, out_dicom_dir, enable_thru_plane: bool = False
    ):
        """
        Save pixel data to the output directory
        based on a reference dicom dataset list
        :param dict_pixel_data: dict of array of data by frame to save to output directory
        :param out_dicom_dir: path to output directory
        :return: None
        """
        # save data frame by frame but with the same series UID and a shared UID pool
        # create output directory
        out_dicom_dir = os.path.join(
            out_dicom_dir,
            list(self._input_datasets.values())[0][0].SeriesInstanceUID,
        )
        os.makedirs(out_dicom_dir, exist_ok=True)  # do not complain if exists

        # generate a new series UID
        series_uid = pydicom_utils.generate_uid()
        uid_pool = set()
        uid_pool.add(series_uid)

        i_instance = 1
        for iframe, frame_seq_name in enumerate(self._input_datasets.keys()):
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
                        for ds in self._input_datasets[frame_seq_name]
                    ]
                    i_ref = distance_to_slices.index(min(distance_to_slices))
                    out_dataset = self._input_datasets[frame_seq_name][i_ref]

                    # set slice specific metadata
                    out_dataset.SliceLocation = slice_loc
                    out_dataset.ImagePositionPatient[0] = str(ipp[0])
                    out_dataset.ImagePositionPatient[1] = str(ipp[1])
                    out_dataset.ImagePositionPatient[2] = str(ipp[2])
                    out_dataset.InstanceNumber = instance_number
                    out_dataset.SliceThickness = slice_thickness

                else:
                    out_dataset = self._input_datasets[frame_seq_name][i_slice]

                # add model and app info to output dicom file
                for tag in self.private_tag:
                    out_dataset.add_new(*tag)

                # set series-wide metadata
                out_dataset.SeriesDescription = "{}{}_{}".format(
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
