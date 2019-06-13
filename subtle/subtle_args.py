#!/usr/bin/env python
'''
args.py

Argument Parser for SubtleGad

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/12/04
'''

import configargparse as argparse

def parser(usage_str, description_str):
    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## shared args
    parser.add_argument('--config', is_config_file=True, help='config file path', default=False)
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='verbose')
    parser.add_argument('--gpu', action='store', dest='gpu_device', type=str, help='set GPU', default=None)
    parser.add_argument('--keras_memory', action='store', dest='keras_memory', type=float, help='set Keras memory (0 to 1)', default=1.)
    parser.add_argument('--checkpoint', action='store', dest='checkpoint_file', type=str, help='checkpoint file', default=None)
    parser.add_argument('--learn_residual', action='store_true', dest='residual_mode', help='learn residual, (zero, low - zero, full - zero)', default=False)
    parser.add_argument('--learning_rate', action='store', dest='lr_init', type=float, help='intial learning rate', default=.001)
    parser.add_argument('--batch_norm', action='store_true', dest='batch_norm', help='batch normalization')
    parser.add_argument('--use_multiprocessing', action='store_true', dest='use_multiprocessing', help='use multiprocessing in generator', default=False)
    parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, help='number of workers for generator', default=1)
    parser.add_argument('--max_queue_size', action='store', dest='max_queue_size', type=int, help='generator queue size', default=16)
    parser.add_argument('--id', action='store', dest='job_id', type=str, help='job id for logging', default='')
    parser.add_argument('--slices_per_input', action='store', dest='slices_per_input', type=int, help='number of slices per input (2.5D)', default=1)
    parser.add_argument('--num_channel_first', action='store', dest='num_channel_first', type=int, help='first layer channels', default=32)
    parser.add_argument('--gen_type', action='store', dest='gen_type', type=str, help='generator type (legacy or split)', default='legacy')
    parser.add_argument('--ssim_lambda', action='store', type=float, dest='ssim_lambda', help='include ssim loss with weight ssim_lambda', default=0.)
    parser.add_argument('--l1_lambda', action='store', type=float, dest='l1_lambda', help='include L1 loss with weight l1_lambda', default=1.)

    ## inference args
    parser.add_argument('--data_preprocess', action='store', dest='data_preprocess', type=str, help='load already-preprocessed data', default=False)
    parser.add_argument('--series_num', action='store', dest='series_num', type=str, help='series number', default='')
    parser.add_argument('--description', action='store', dest='description', type=str, help='append to end of series description', default='')
    parser.add_argument('--path_out', action='store', dest='path_out', type=str, help='path to output SubtleGad dicom dir', default=None)
    parser.add_argument('--path_zero', action='store', dest='path_zero', type=str, help='path to zero dose dicom dir', default=None)
    parser.add_argument('--path_low', action='store', dest='path_low', type=str, help='path to low dose dicom dir', default=None)
    parser.add_argument('--path_full', action='store', dest='path_full', type=str, help='path to full dose dicom dir', default=None)
    parser.add_argument('--path_base', action='store', dest='path_base', type=str, help='path to base dicom directory containing subdirs', default=None)
    parser.add_argument('--discard_start_percent', action='store', type=float, dest='discard_start_percent', help='throw away start X %% of slices', default=0.)
    parser.add_argument('--discard_end_percent', action='store', type=float, dest='discard_end_percent', help='throw away end X %% of slices', default=0.)
    parser.add_argument('--mask_threshold', action='store', type=float, dest='mask_threshold', help='cutoff threshold for mask', default=.08)
    parser.add_argument('--transform_type', action='store', type=str, dest='transform_type', help="transform type ('rigid', 'translation', etc.)", default='rigid')
    parser.add_argument('--normalize', action='store_true', dest='normalize', help="global scaling", default=False)
    parser.add_argument('--scale_matching', action='store_true', dest='scale_matching', help="match scaling of each image to each other", default=False)
    parser.add_argument('--joint_normalize', action='store_true', dest='joint_normalize', help="use same global scaling for all images", default=False)
    parser.add_argument('--normalize_fun', action='store', dest='normalize_fun', type=str, help='normalization fun', default='mean')
    parser.add_argument('--skip_registration', action='store_true', dest='skip_registration', help='skip co-registration', default=False)
    parser.add_argument('--skip_mask', action='store_true', dest='skip_mask', help='skip mask', default=False)
    parser.add_argument('--skip_scale_im', action='store_true', dest='skip_scale_im', help='skip histogram matching', default=False)
    parser.add_argument('--scale_dicom_tags', action='store_true', dest='scale_dicom_tags', help='use dicom tags for relative scaling', default=False)


    ## train args
    parser.add_argument('--data_list', action='store', dest='data_list_file', type=str, help='list of pre-processed files for training', default=None)
    parser.add_argument('--data_dir', action='store', dest='data_dir', type=str, help='location of data', default=None)
    parser.add_argument('--file_ext', action='store', dest='file_ext', type=str, help='file extension of data', default=None)
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
    parser.add_argument('--predict_file_ext', action='store', dest='predict_file_ext', type=str, help='file extension of predcited data', default='npy')
    parser.add_argument('--no_save_best_only', action='store_false', dest='save_best_only', default=True, help='save newest model at every checkpoint')
    parser.add_argument('--zoom', action='store', dest='zoom', type=int, help='zoom to in-plane matrix size', default=None)
    parser.add_argument('--zoom_order', action='store', dest='zoom_order', type=int, help='zoom order', default=3)
    parser.add_argument('--nslices', action='store', dest='nslices', type=int, help='number of slices for scaling', default=20)
    parser.add_argument('--global_scale_ref_im0', action='store_true', dest='global_scale_ref_im0', help="use zero-dose for global scaling ref", default=False)
    parser.add_argument('--override_dicom_naming', action='store_true', dest='override', help='dont check dicom names', default=False)
    parser.add_argument('--denoise', action='store_true', dest='denoise', help='denoise lowcon', default=False)

    parser.add_argument('--input_idx', nargs='+', type=int, help='input indices from data', default=[0, 1])
    parser.add_argument('--output_idx', nargs='+', type=int, help='output indices from data', default=[2])

    return parser

