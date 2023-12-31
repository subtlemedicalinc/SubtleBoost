#!/usr/bin/env python
'''
args.py

Argument Parser for SubtleGad

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/12/04
'''

import configargparse as argparse

def _model_arch_args(parser):
    parser.add_argument('--model_name', action='store', dest='model_name', type=str, help='Name of the model architecture to be trained/tested on, Ex: unet2d, mres2d', default='unet2d')
    parser.add_argument('--model_config', action='store', dest='model_config', type=str, help='Name of the model config to use from the defined JSON file', default='base')

    return parser

def _breast_processing(parser):
    parser.add_argument('--breast_gad', action='store_true', dest='breast_gad', help='If true, breast specific pre-processing is done')
    parser.add_argument('--breast_mask_path', action='store', dest='breast_mask_path', type=str, help='Path to where breast masks are stored', default='')
    parser.add_argument('--external_reg_ref', action='store', dest='external_reg_ref', type=str, help='Path to series for external registration reference')

    return parser

def _shared_args(parser):
    parser.add_argument('--experiment', action='store', dest='experiment', type=str, help='Name of the experiment for which preprocess/train/inference is to be run', default=None)
    parser.add_argument('--sub_experiment', action='store', dest='sub_experiment', type=str, help='Name of the sub experiment for which preprocess/train/inference is to be run', default=None)
    parser.add_argument('--hypsearch_name', action='store', dest='hypsearch_name', type=str, help='Name of the hyperparameter search experiment', default=None)

    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')
    parser.add_argument('--gpu', action='store', dest='gpu', type=str, help='set GPU', default=None)
    parser.add_argument('--use_multiprocessing', action='store_true', dest='use_multiprocessing', help='use multiprocessing in generator', default=False)
    parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, help='number of workers for generator', default=8)
    parser.add_argument('--procs_per_gpu', action='store', dest='procs_per_gpu', type=int, help='For parallel MPR processing, this number specifies the number of processes to be run on a single GPU', default=1)
    parser.add_argument('--max_queue_size', action='store', dest='max_queue_size', type=int, help='generator queue size', default=16)
    parser.add_argument('--id', action='store', dest='job_id', type=str, help='job id for logging', default='')
    parser.add_argument('--slices_per_input', action='store', dest='slices_per_input', type=int, help='number of slices per input (2.5D)', default=1)
    parser.add_argument('--gen_type', action='store', dest='gen_type', type=str, help='generator type (legacy or split)', default='legacy')
    parser.add_argument('--path_out', action='store', dest='path_out', type=str, help='path to output SubtleGad dicom dir', default=None)
    parser.add_argument('--path_zero', action='store', dest='path_zero', type=str, help='path to zero dose dicom dir', default=None)
    parser.add_argument('--path_low', action='store', dest='path_low', type=str, help='path to low dose dicom dir', default=None)
    parser.add_argument('--path_full', action='store', dest='path_full', type=str, help='path to full dose dicom dir', default=None)
    parser.add_argument('--path_base', action='store', dest='path_base', type=str, help='path to base dicom directory containing subdirs', default=None)

    ## 3D patch based
    parser.add_argument('--block_size', action='store', type=int, dest='block_size', help='Block size for 3D patch based training', default=64)
    parser.add_argument('--block_strides', action='store', type=int, dest='block_strides', help='Block strides for 3D patch based training', default=32)

    parser.add_argument('--input_idx', type=str, help='input indices from data', default='0,1')
    parser.add_argument('--output_idx', type=str, help='output indices from data', default='2')
    parser.add_argument('--slice_axis', action='store',  type=int, dest='slice_axis',  help='axes for slice direction', default=0)

    parser.add_argument('--acq_plane', action='store', type=str, dest='acq_plane', help='Plane of acquisition - AX, SAG or COR', default='AX')

    return parser

def _multi_contrast_args(parser):
    parser.add_argument('--multi_contrast_mode', action='store_true', dest='multi_contrast_mode', help='T2 or FLAIR preprocessing mode', default=False)
    parser.add_argument('--multi_contrast_kw', action='store', type=str, dest='multi_contrast_kw', help='Comma separated keywords for identifying multicontrast sequences', default=None)

    # This scaling constant is computed using the maximum intensity between post contrast
    # and low-dose T1 images. It is the average of post_contrast.max() / low_dose.max()
    # of all the Stanford GE cases. This might vary between manufacturers or sites
    # The T2 volume's histogram is equalized using T1 low-dose as reference and is
    # multiplied with this constant value
    parser.add_argument('--t2_scaling_constant', action='store', type=float,
    dest='t2_scaling_constant', help='Intensity scale factor that the T2 volume should be scaled with', default=1.389)

    return parser

def _uad_args(parser):
    parser.add_argument('--enh_mask_uad', action='store_true', dest='enh_mask_uad',
    help='Boolean arg indicating whether to use enhancement weighted loss function from the UAD masks', default=False)

    parser.add_argument('--uad_mask_path', action='store', type=str, dest='uad_mask_path',
    help='Basepath where UAD masks are stored', default=None)

    parser.add_argument('--uad_mask_threshold', action='store', type=float,
    dest='uad_mask_threshold', help='Percentage of max pixel value to compute the UAD mask',
    default=0.1)

    parser.add_argument('--uad_file_ext', action='store', type=str, dest='uad_file_ext',
    help='File extension of UAD mask files', default=None)

    parser.add_argument('--use_uad_ch_input', action='store_true', dest='use_uad_ch_input', help='If True, uad mask will be used as a separate input channel', default=False)

    parser.add_argument('--uad_ip_channels', action='store', type=int, dest='uad_ip_channels', help='Number of channels of UAD input', default=1)

    return parser


def _preprocess_args(parser):
    parser.add_argument('--discard_start_percent', action='store', type=float, dest='discard_start_percent', help='throw away start X %% of slices', default=0.)
    parser.add_argument('--discard_end_percent', action='store', type=float, dest='discard_end_percent', help='throw away end X %% of slices', default=0.)
    parser.add_argument('--mask_threshold', action='store', type=float, dest='mask_threshold', help='cutoff threshold for mask', default=.08)
    parser.add_argument('--noise_mask_area', action='store_true', dest='noise_mask_area', help="If True, region with the largest area will be picked as a noise mask after performing connected components", default=False)
    parser.add_argument('--transform_type', action='store', type=str, dest='transform_type', help="transform type ('rigid', 'translation', etc.)", default='rigid')
    parser.add_argument('--register_with_dcm_reference', action='store_true', dest='register_with_dcm_reference', help="If true, a SITK instance derived from the DCM images will be passed as reference. This helps in passing the exact image origin, cosine direction etc.,", default=False)
    parser.add_argument('--reg_n_levels', type=int, action='store', dest='reg_n_levels', help='Number of levels of affine registration', default=4)
    parser.add_argument('--normalize', action='store_true', dest='normalize', help="global scaling", default=False)
    parser.add_argument('--scale_matching', action='store_true', dest='scale_matching', help="match scaling of each image to each other", default=False)
    parser.add_argument('--joint_normalize', action='store_true', dest='joint_normalize', help="use same global scaling for all images", default=False)
    parser.add_argument('--normalize_fun', action='store', dest='normalize_fun', type=str, help='normalization fun', default='mean')
    parser.add_argument('--skip_registration', action='store_true', dest='skip_registration', help='skip co-registration', default=False)
    parser.add_argument('--skip_mask', action='store_true', dest='skip_mask', help='skip mask', default=False)
    parser.add_argument('--skip_hist_norm', action='store_true', dest='skip_hist_norm', help='skip histogram matching', default=False)
    parser.add_argument('--scale_dicom_tags', action='store_true', dest='scale_dicom_tags', help='use dicom tags for relative scaling', default=False)
    parser.add_argument('--zoom', action='store', dest='zoom', type=int, help='zoom to in-plane matrix size', default=None)
    parser.add_argument('--resize', action='store', dest='resize', type=int, help='resize to 3D matrix size', default=None)
    parser.add_argument('--resample_isotropic', action='store', type=float, dest='resample_isotropic', help='resample to 1mm isotropic resolution', default=0)
    parser.add_argument('--resample_size', action='store', dest='resample_size', type=int, help='resample 3D matrix (generally used to train models with downsampled images)', default=None)
    parser.add_argument('--zoom_order', action='store', dest='zoom_order', type=int, help='zoom order', default=3)
    parser.add_argument('--nslices', action='store', dest='nslices', type=int, help='number of slices for scaling', default=20)
    parser.add_argument('--global_scale_ref_im0', action='store_true', dest='global_scale_ref_im0', help="use zero-dose for global scaling ref", default=False)
    parser.add_argument('--override_dicom_naming', action='store_true', dest='override', help='dont check dicom names', default=False)

    # brain masking related args
    # FIXME: change from FSL to a generic name
    parser.add_argument('--fsl_mask', action='store_true', dest='fsl_mask',
                        help='Extract brain using FSL BET', default=False)
    parser.add_argument('--fsl_threshold', action='store', type=float,
                        dest='fsl_threshold', help='Fraction parameter for FSL BET', default=0.5)
    parser.add_argument('--fsl_area_threshold_cm2', action='store', type=float,
                        dest='fsl_area_threshold_cm2', help='Reject slices which have extracted brain area (in cm2) lower than the threshold. If argument is not given, all slices will be included', default=None)
    parser.add_argument('--fsl_mask_all_ims', action='store_true', dest='fsl_mask_all_ims', help='If `fsl_mask`, perform FSL BET on all ims and take the union', default=False)
    parser.add_argument('--use_fsl_reg', action='store_true', dest='use_fsl_reg', help='If true, the registration parameters computed from skull stripped brain will be applied on full brain, otherwise full brain will be registered separately', default=True)
    parser.add_argument('--non_rigid_reg', action='store_true', dest='non_rigid_reg', help='If true, then bspline parameter map is set with affine map to perform non-rigid registration', default=False)
    parser.add_argument('--union_brain_masks', action='store_true', dest='union_brain_masks', help='Specify whether the masks from the three images should be combined using union operation or not', default=False)

    parser.add_argument('--pad_for_size', action='store', type=int,
    dest='pad_for_size', help='If True and if matrix sizes are different then zero padding is done for the final size is equal to this param', default=0)

    parser.add_argument('--save_preprocess_video', action='store_true', dest='save_preprocess_video', help='If True, preprocess videos are saved in MP4 format', default=False)
    parser.add_argument('--data_batch', action='store', dest='data_batch', type=str, help='Argument to preprocess data in batches. This string specified the start index and the number of cases to process in a comma separated fashion. Ex: "0, 10" will process cases starting from index 0 to index 10 as defined in the experiment data.json')
    parser.add_argument('--blur_for_cs_streaks', action='store_true', dest='blur_for_cs_streaks',
    help='If True, then low-dose images are blurred with a one-directional gaussian blur. This is done to get rid of compressed sensing related streaks found in Tiantan Philips low-dose input', default=False)

    return parser

def _train_args(parser):
    parser.add_argument('--keras_memory', action='store', dest='keras_memory', type=float, help='set Keras memory (0 to 1)', default=1.)
    parser.add_argument('--checkpoint', action='store', dest='checkpoint', type=str, help='checkpoint file', default=None)
    parser.add_argument('--resume_from_checkpoint', action='store', dest='resume_from_checkpoint', type=str, help='If specified, training script will resume training from the given checkpoint', default=None)
    parser.add_argument('--learn_residual', action='store_true', dest='residual_mode', help='learn residual, (zero, low - zero, full - zero)', default=False)
    parser.add_argument('--learning_rate', action='store', dest='lr_init', type=float, help='intial learning rate', default=.001)
    parser.add_argument('--optimizer', action='store', dest='optimizer', type=str, help='Optimizer used to train the model', default='adam')
    parser.add_argument('--optim_amsgrad', action='store_true', dest='optim_amsgrad', help='AMS grad for Adam optimizer', default=True)


    # training loss weights
    parser.add_argument('--ssim_lambda', action='store', type=float, dest='ssim_lambda', help='include ssim loss with weight ssim_lambda', default=0.)
    parser.add_argument('--l1_lambda', action='store', type=float, dest='l1_lambda', help='include L1 loss with weight l1_lambda', default=1.)
    parser.add_argument('--perceptual_lambda', action='store', type=float, dest='perceptual_lambda', help='Loss from VGG19 ImageNet model', default=0.)
    parser.add_argument('--vgg19_ckp', action='store', type=str, dest='vgg19_ckp', help='Checkpoint path of customized pretrained VGG 19 imagenet weights imported from tensorflow', default=None)
    parser.add_argument('--wloss_lambda', action='store', type=float, dest='wloss_lambda', help='Wasserstein loss', default=0.)
    parser.add_argument('--style_lambda', action='store', type=float, dest='style_lambda', help='Style or texture loss lambda', default=0.)
    parser.add_argument('--vgg_resize_shape', action='store', type=int, dest='vgg_resize_shape',
    help='Resize input tensors to this size before computing the VGG MSE loss.', default=0)

    parser.add_argument('--data_list', action='store', dest='data_list_file', type=str, help='list of pre-processed files for training', default=None)
    parser.add_argument('--data_dir', action='store', dest='data_dir', type=str, help='location of data', default=None)
    parser.add_argument('--file_ext', action='store', dest='file_ext', type=str, help='file extension of data', default='npy')
    parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs to run', default=10)
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--tbimage_batch_size', action='store', dest='tbimage_batch_size', type=int, help='TBImage batch size', default=8)
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
    parser.add_argument('--validation_split', action='store', dest='validation_split', type=float, help='ratio of validation data', default=.1)
    parser.add_argument('--log_dir', action='store', dest='log_dir', type=str, help='log directory', default='logs')
    parser.add_argument('--max_data_sets', action='store', dest='max_data_sets', type=int, help='limit number of data sets', default=None)
    parser.add_argument('--predict', action='store', dest='predict_dir', type=str, help='perform prediction and write to directory', default=None)
    parser.add_argument('--steps_per_epoch', action='store', dest='steps_per_epoch', type=int, help='number of iterations per epoch (default -- # slices in dataset / batch_size', default=None)
    parser.add_argument('--val_steps_per_epoch', action='store', dest='val_steps_per_epoch', type=int, help='# slices in dataset / batch_size', default=None)
    parser.add_argument('--shuffle', action='store_true', dest='shuffle', help='shuffle input data files each epoch', default=False)
    parser.add_argument('--positive_only', action='store_true', dest='positive_only', help='keep only positive part of residual', default=False)
    parser.add_argument('--history_file', action='store', dest='history_file', type=str, help='store history in npy file', default=None)
    parser.add_argument('--stats_file', action='store', dest='stats_file', type=str, help='store inference stats in h5 file', default=None)
    parser.add_argument('--predict_file_ext', action='store', dest='predict_file_ext', type=str, help='file extension of predcited data', default='npy')
    parser.add_argument('--no_save_best_only', action='store_false', dest='save_best_only', default=True, help='save newest model at every checkpoint')
    parser.add_argument('--save_all_weights', action='store_true', dest='save_all_weights',
    default=False, help='If True, then weights of all epochs will be saved in the checkpoint dir')

    parser.add_argument('--denoise', action='store_true', dest='denoise', help='denoise lowcon', default=False)
    parser.add_argument('--enh_mask', action='store_true', dest='enh_mask', help='If True, then enhancement_mask is computed and used to compute L1 loss', default=False)
    parser.add_argument('--enh_pfactor', action='store', dest='enh_pfactor', type=float, help='The power factor in the term to compute smooth enhancement mask', default=1.0)

    parser.add_argument('--enh_mask_t2', action='store_true', dest='enh_mask_t2', help='If enh_mask is True, specifying true in this arg, will make the enhancement mask computation, use the T2 volume', default=False)

    parser.add_argument('--train_mpr', action='store_true', dest='train_mpr', help='train acrossa multiple planes ', default=False)
    parser.add_argument('--reshape_for_mpr_rotate', action='store_true',
    dest='reshape_for_mpr_rotate', help='If True, angle rotation is done without cropping, thus making the model input different for each angle', default=False)

    parser.add_argument('--brain_only', action='store_true', dest='brain_only', help='Use FSL extracted brain data to train (preprocess should have been run with this option: H5 file should have "data_mask" key)', default=False)
    parser.add_argument('--brain_only_mode', action='store', dest='brain_only_mode', type=str, help='pure or mixed - whether to train only on FSL masked images only or include a fraction of the full brain images too', default=None)

    parser.add_argument('--pretrain_ckps', action='store', dest='pretrain_ckps', type=str, help='List of pretrained checkpoints from which weights should be loaded to different portions of the network. Used in GeneratorFBoostUNet2D', default=None)

    # gan related
    parser.add_argument('--gan_mode', action='store_true', dest='gan_mode', help='If True, network will be trained in GAN mode with adversarial loss', default=False)
    parser.add_argument('--adversary_name', action='store', dest='adversary_name', type=str, help='Name of the Discriminator model architecture', default='patch2d')
    parser.add_argument('--num_disc_steps', action='store', dest='num_disc_steps', type=int, help='Number of steps to train Discriminator for, for every generator epoch', default=5)
    parser.add_argument('--disc_lr_init', action='store', dest='disc_lr_init', type=float, help='Learning rate initialization for adversary Adam optimizer', default=2e-3)
    parser.add_argument('--disc_beta', action='store', dest='disc_beta', type=float, help='Beta value for adversary Adam optimizer', default=0.5)
    parser.add_argument('--disc_loss_function', action='store', type=str, dest='disc_loss_function', help='Loss function for adversary', default='mse')
    parser.add_argument('--add_disc_noise', action='store_true', dest='add_gen_noise', help='If True, random noise will be added to discriminator input', default=False)

    parser.add_argument('--multi_slice_gt', action='store_true', dest='multi_slice_gt', help='If True, then in 2.5D training, ground truth will also be 7 slices', default=False)

    return parser

def _inference_args(parser):
    parser.add_argument('--config', type=str, dest='config', help='Path to inference config file')
    parser.add_argument('--num_channel_output', type=int, dest='num_channel_output', help='Number of channels output for the model', default=1)
    parser.add_argument('--dcm_pre', type=str, dest='dcm_pre', help='DICOM path for pre-contrast')
    parser.add_argument('--dcm_post', type=str, dest='dcm_post', help='DICOM path for post-contrast or low-dose')
    parser.add_argument('--out_folder', action='store', dest='out_folder', type=str, help='Output folder name for inference pipeline')
    parser.add_argument('--predict_full_volume', action='store_true', dest='predict_full_volume', help='If true, then inference is run on whole volume', default=False)
    parser.add_argument('--data_preprocess', action='store', dest='data_preprocess', type=str, help='load already-preprocessed data', default=False)
    parser.add_argument('--dicom_inference', action='store_true', dest='dicom_inference', help='execute inference pipeline directly from dicom files', default=False)
    parser.add_argument('--series_num', action='store', dest='series_num', type=str, help='series number', default='')
    parser.add_argument('--description', action='store', dest='description', type=str, help='append to end of series description', default='')
    parser.add_argument('--inference_mpr', action='store_true', dest='inference_mpr', help='run through multiple planes and average', default=False)
    parser.add_argument('--inference_mpr_avg', action='store', dest='inference_mpr_avg', type=str, help='type of MPR averaging', default='mean')
    parser.add_argument('--undo_pad_resample', action='store', type=str,
    dest='undo_pad_resample', help='Applicable only for inference. Format (crop_x, crop_y) to undo the zero padding done for uniformity', default=None)
    parser.add_argument('--num_rotations', action='store', dest='num_rotations', type=int, help='number of rotations to average', default=1)
    parser.add_argument('--match_scales_fsl', action='store_true', dest='match_scales_fsl', help='If True, the full head image dynamic range is matched with the FSL extracted dynamic range. This should be set to True, only when the model checkpoint used here was trained with only the FSL masked brain region', default=False)

    parser.add_argument('--brain_centering', action='store_true', dest='brain_centering', help='Vertical and horizontal centering of the brain', default=False)

    return parser

def get_parser(usage_str='', description_str=''):
    parser = argparse.ArgumentParser(
        usage=usage_str,
        description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser = _model_arch_args(parser)
    parser = _shared_args(parser)
    parser = _preprocess_args(parser)
    parser = _train_args(parser)
    parser = _inference_args(parser)
    parser = _multi_contrast_args(parser)
    parser = _uad_args(parser)
    parser = _breast_processing(parser)

    return parser
