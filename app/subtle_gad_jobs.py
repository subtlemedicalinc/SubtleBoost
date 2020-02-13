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
from deepbrain import Extractor as BrainExtractor

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
            "normalize": True,
            "normalize_fun": "mean",

            "perform_noise_mask": True,
            "noise_mask_threshold": 0.08,
            "noise_mask_area": False,
            "noise_mask_selem": False,

            "skull_strip": True,
            "skull_strip_union": True,
            "skull_strip_prob_threshold": 0.5,

            "perform_dicom_scaling": False,
            "transform_type": "rigid",
            "histogram_matching": True,
            "num_scale_context_slices": 20,
            "joint_normalize": False,
            "scale_ref_zero_img": False,

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
        BaseJobType.__init__(self, task, model_dir)

        self._logger.info("Loading model from %s...", model_dir)
        self._model = GenericInferenceModel(model_dir=model_dir,
                                            decrypt_key_hex=decrypt_key_hex)
        self._logger.info("Loaded %s model.", self._model.model_type)
        if self._model.model_type == 'invalid':
            raise NotImplementedError("Invalid model found in {}".format(model_dir))
        # update model config with app config:
        # to update tunable parameters, job parameters and app parameters
        self._model.update_config(task.job_definition.exec_config)

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

        # initialize dict to save intermediate values and flags
        self._metadata = {}

        # list of lambda functions constructed during preprocess
        self._proc_lambdas = []

        self._output_data = {}

    def _preprocess(self):
        """The preprocess func

        :param _: (in_dicom in BaseJobType by default)
        """
        self._get_dicom_data()

        self._get_pixel_spacing()
        self._get_dicom_scale_coeff()
        #
        # get original pixel data and meta data info
        self._get_raw_pixel_data()

        # dictionary of the data to process by frame
        dict_pixel_data = self._preprocess_raw_pixel_data()

        return dict_pixel_data

    # pylint: disable=arguments-differ
    def _process(self, dict_pixel_data):
        """Process the pixel data with the meta data.

        :param dict_pixel_data: dictionary of the input pixel data (numpy arrays) by frame
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

        self._input_series = (zero_dose_series, low_dose_series)

        self._input_datasets = (
            zero_dose_series.get_dict_sorted_datasets(),
            low_dose_series.get_dict_sorted_datasets()
        )

    def _get_raw_pixel_data(self):
        for frame_seq_name, _ in self._input_datasets[0].items():
            zero_data_np = self._input_series[0].get_pixel_data()[frame_seq_name]
            low_data_np = self._input_series[1].get_pixel_data()[frame_seq_name]

            self._raw_input[frame_seq_name] = np.array([zero_data_np, low_data_np])

            self._logger.info(
                "the shape of array {}".format(
                    self._raw_input[frame_seq_name].shape
                )
            )

    def _mask_noise(self, images):
        mask = []

        if self._proc_config.perform_noise_mask:
            self._logger.info('Performing noise masking...')

            threshold = self._proc_config.noise_mask_threshold
            mask_area = self._proc_config.noise_mask_area
            use_selem = self._proc_config.noise_mask_selem

            mask_fn = lambda idx: (lambda ims: ims[idx] * \
            preprocess_single_series.mask_bg_noise(ims[idx], threshold, mask_area, use_selem))

            for idx in range(images.shape[0]):
                noise_mask = mask_fn(idx)(images)
                images[idx, ...] *= noise_mask
                mask.append(noise_mask)

            self._proc_lambdas.append({
                'name': 'noise_mask',
                'fn': [mask_fn(0), mask_fn(1)]
            })

        return images, np.array(mask)

    def _strip_skull_npy_vol(self, img_npy):
        brain_ext = BrainExtractor()

        img_scaled = np.interp(img_npy, (img_npy.min(), img_npy.max()), (0, 1))
        segment_probs = brain_ext.run(img_scaled)

        th = self._proc_config.skull_strip_prob_threshold
        return preprocess_single_series.get_largest_connected_component(segment_probs > th)

    def _strip_skull(self, images):
        brain_mask = None

        if self._proc_config.skull_strip:
            self._logger.info('Performing skull stripping for zero dose...')
            mask_zero = self._strip_skull_npy_vol(images[0])

            if self._proc_config.skull_strip_union:
                self._logger.info('Performing skull stripping for low dose...')
                mask_low = self._strip_skull_npy_vol(images[1])
                brain_mask = ((mask_zero > 0) | (mask_low > 0))
            else:
                brain_mask = mask_zero

        return brain_mask

    def _apply_brain_mask(self, images, brain_mask):
        self._logger.info('Applying computed brain mask on images')
        masked_images = np.copy(images)

        masked_images[0] *= brain_mask
        masked_images[1] *= brain_mask
        return masked_images

    def _has_dicom_scaling_info(self):
        header_zero, _ = self._get_dicom_header()
        return 'RescaleSlope' in header_zero

    def _get_dicom_header(self):
        for frame_seq_name, _ in self._input_datasets[0].items():
            header_zero = self._input_datasets[0][frame_seq_name][0]
            header_low = self._input_datasets[1][frame_seq_name][0]

            return (header_zero, header_low)

    def _get_dicom_scale_coeff(self):
        if not self._has_dicom_scaling_info():
            self._dicom_scale_coeff = {
                'zero': {
                    'rescale_slope': 1.0,
                    'rescale_intercept': 0.0,
                    'scale_slope': 0.0
                },
                'low': {
                    'rescale_slope': 1.0,
                    'rescale_intercept': 0.0,
                    'scale_slope': 0.0
                }
            }

        else:
            header_zero, header_low = self._get_dicom_header()

            self._dicom_scale_coeff = [{
                'rescale_slope': float(header_zero.RescaleSlope),
                'rescale_intercept': float(header_zero.RescaleIntercept),
                'scale_slope': float(header_zero[0x2005, 0x100e].value)
            }, {
                'rescale_slope': float(header_low.RescaleSlope),
                'rescale_intercept': float(header_low.RescaleIntercept),
                'scale_slope': float(header_low[0x2005, 0x100e].value)
            }]

    @staticmethod
    def _scale_slope_intercept(img, rescale_slope, rescale_intercept, scale_slope):
        return (img * rescale_slope + rescale_intercept) / (rescale_slope * scale_slope)

    def _scale_intensity_with_dicom_tags(self, images):
        if self._proc_config.perform_dicom_scaling and self._has_dicom_scaling_info():
            self._logger.info('Performing intensity scaling using DICOM tags...')

            scaled_images = np.copy(images)

            scale_fn = lambda idx: (
                lambda ims: SubtleGADJobType._scale_slope_intercept(ims[idx],
                **self._dicom_scale_coeff[idx])
            )

            scaled_images[0] = scale_fn(0)(scaled_images)
            scaled_images[1] = scale_fn(1)(scaled_images)

            self._proc_lambdas.append({
                'name': 'dicom_scaling',
                'fn': [scale_fn(0), scale_fn(1)]
            })

            return scaled_images

        return images

    def _get_pixel_spacing(self):
        header_zero, header_low = self._get_dicom_header()

        x_zero, y_zero = header_zero.PixelSpacing
        z_zero = header_zero.SliceThickness

        x_low, y_low = header_low.PixelSpacing
        z_low = header_low.SliceThickness

        self._pixel_spacing = [(x_zero, y_zero, z_zero), (x_low, y_low, z_low)]

    def _register(self, images):
        self._logger.info('Performing registration of low dose with respect to zero dose...')

        reg_images = np.copy(images)

        # register the low dose image with respect to the zero dose image
        reg_images[1], reg_params = preprocess_single_series.register(
            fixed_img=reg_images[0], moving_img=reg_images[1],
            fixed_spacing=self._pixel_spacing[0], moving_spacing=self._pixel_spacing[1],
            transform_type=self._proc_config.transform_type
        )

        idty_fn = lambda ims: ims[0]

        apply_reg = lambda ims: preprocess_single_series.apply_registration(ims[1], \
        self._pixel_spacing[1], reg_params)

        self._proc_lambdas.append({
            'name': 'register',
            'fn': [idty_fn, apply_reg]
        })

        return reg_images

    def _match_histogram(self, images):
        if self._proc_config.histogram_matching:
            self._logger.info('Matching histogram of low dose with respect to zero dose...')
            hist_images = np.copy(images)

            hist_images[1] = preprocess_single_series.match_histogram(
                img=hist_images[1], ref_img=hist_images[0]
            )

            idty_fn = lambda ims: ims[0]
            hist_match_fn = lambda ims: preprocess_single_series.match_histogram(ims[1], ims[0])

            self._proc_lambdas.append({
                'name': 'histogram_matching',
                'fn': [idty_fn, hist_match_fn]
            })

            return hist_images

        return images

    def _scale_intensity(self, images, noise_mask):
        scale_images = np.copy(images)

        self._logger.info('Performing intensity scaling...')
        num_slices = scale_images.shape[1]

        idx_center = range(
            num_slices//2 - self._proc_config.num_scale_context_slices//2,
            num_slices//2 + self._proc_config.num_scale_context_slices//2
        )

        # use pixels inside the noise mask of zero dose

        ref_mask = noise_mask[0, idx_center]
        
        context_img_zero = scale_images[0, idx_center, ...][ref_mask != 0].ravel()
        context_img_low = scale_images[1, idx_center, ...][ref_mask != 0].ravel()

        context_img = np.stack((context_img_zero, context_img_low), axis=0)

        scale_factor = preprocess_single_series.get_intensity_scale(
            img=context_img[0], ref_img=context_img[1], levels=np.linspace(.5, 1.5, 30)
        )

        self._logger.info('Computed intensity scale factor is %s', scale_factor)

        scale_images[1] *= scale_factor
        context_img[1] *= scale_factor

        sfactors = [1, scale_factor]
        match_scales_fn = lambda idx: (lambda ims: ims[idx] * sfactors[idx])
        self._proc_lambdas.append({
            'name': 'match_scales',
            'fn': [match_scales_fn(0), match_scales_fn(1)]
        })

        context_img = context_img.transpose(1, 0)
        scale_images = scale_images.transpose(1, 0, 2, 3)

        print('context img', context_img.shape)
        print('scale images', scale_images.shape)

        norm_axis = (0, 1) if self._proc_config.joint_normalize else (0, )

        if self._proc_config.scale_ref_zero_img:
            norm_img = context_img[..., 0]
            norm_axis = (0, )
        else:
            norm_img = context_img

        print('norm image shape', norm_img.shape)
        print('norm axis', norm_axis)

        global_scale = np.mean(norm_img, axis=norm_axis)
        for a in norm_axis:
            global_scale = np.expand_dims(global_scale, axis=a)

        # repeat the computed scale such that its shape is always (1, 2)
        if global_scale.ndim == 1:
            global_scale = np.repeat([global_scale], repeats=scale_images.shape[1], axis=1)
        elif global_scale.ndim == 2 and global_scale.shape[1] == 1:
            global_scale = np.repeat(global_scale, repeats=scale_images.shape[1], axis=1)

        global_scale = global_scale[:, :, None, None]
        self._logger.info('Computed global scale %s with shape %s', global_scale, \
        global_scale.shape)

        scale_images /= global_scale
        global_scale_fn = lambda idx: (
            lambda ims: (ims[idx] / global_scale[:, idx])
        )

        self._proc_lambdas.append({
            'name': 'global_scale',
            'fn': [global_scale_fn(0), global_scale_fn(1)]
        })

        return scale_images.transpose(1, 0, 2, 3)

    def _apply_proc_lambdas(self, unmasked_data):
        self._logger.info('Applying all preprocessing steps on full brain images...')
        processed_data = np.copy(unmasked_data)

        for proc_lambda in self._proc_lambdas:
            self._logger.info('::APPLY PROC LAMBDAS::%s', proc_lambda['name'])

            for idx, fn in enumerate(proc_lambda['fn']):
                processed_data[idx] = fn(processed_data)

        return processed_data

    def _preprocess_raw_pixel_data(self):
        """
        get pixel data from dicom datasets
        and resample the pixel data to the specified shape
        :return: preprocessed pixel data
        """
        # apply preprocessing for each frame of the dictionary
        dict_input_data = {}

        for frame_seq_name, data_array in self._raw_input.items():
            input_data_full = data_array.copy()

            # write preprocess chain here
            input_data_mask, noise_mask = self._mask_noise(input_data_full)

            brain_mask = self._strip_skull(input_data_mask)
            input_data_mask = self._apply_brain_mask(input_data_mask, brain_mask)

            input_data_mask = self._scale_intensity_with_dicom_tags(input_data_mask)
            input_data_mask = self._register(input_data_mask)
            input_data_mask = self._match_histogram(input_data_mask)
            input_data_mask = self._scale_intensity(input_data_mask, noise_mask)

            input_data_full = self._apply_proc_lambdas(input_data_full)
            np.save('/home/srivathsa/app_output/debug/final.npy', input_data_full)

            # TEMP
            input_data = np.array([input_data_full[0, 180:194]]).transpose(0, 2, 3, 1)

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
