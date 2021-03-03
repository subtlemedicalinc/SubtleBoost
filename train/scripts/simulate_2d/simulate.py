#!/usr/bin/env python

'''
simulate.py

Simulate 2D Brain MRI images with thick slices from 3D thin slices. Optionally augment the
datasets with ringing and/or motion artifacts.

@author: Srivathsa Pasumarthi (srivathsa@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2021/02/05
'''

import os
import shutil
import argparse
from glob import glob
from tqdm import tqdm
import json
import uuid
import numpy as np
import pydicom
import torchio as tio
import pandas as pd

import subtle.utils.io as suio
from QC.util.aliasing_sim import AliasingLayer

usage_str = 'usage: %(prog)s [options]'
description_str = 'generate 2D image with motion artifacts simulation'

T1_KW = 'bravo'

def generate_uuid():
    prefix = "1.2.826.0.1.3680043.10.221."
    entropy_src = uuid.uuid4().int
    avail_digits = 64 - len(prefix)
    int_val = entropy_src % (10 ** avail_digits)
    return prefix + str(int_val)

def rename_dcm_meta(dirpath_series, study_uid):
    all_dcms = glob('{}/*.dcm'.format(dirpath_series), recursive=True)
    case_num = dirpath_series.split('/')[-2]

    for fpath_dcm in tqdm(all_dcms, total=len(all_dcms)):
        dcm_ds = pydicom.dcmread(fpath_dcm)
        dcm_ds.PatientID = case_num
        dcm_ds.PatientsName = case_num
        dcm_ds.PatientName = case_num
        dcm_ds.StudyInstanceUID = study_uid

        pydicom.dcmwrite(fpath_dcm, dcm_ds)

def case_num_transform(case_str):
    cnum = int(case_str.split('_')[-1])
    return 'P{0:04d}_2DSIM'.format(cnum)

def get_pixel_spacing(hdr):
    return np.array([float(hdr.PixelSpacing[0]), float(hdr.PixelSpacing[1]), float(hdr.SliceThickness)])

def slice_avg_3d_vol(fpath_input):
    dcm_vol, hdr = suio.dicom_files(fpath_input)
    or_sl_thk = hdr.SpacingBetweenSlices
    num_sl = int(np.ceil(args.slice_thk / or_sl_thk))

    dcm_vol_crop = dcm_vol[args.skip_idx:-args.skip_idx]
    mod_sl = dcm_vol_crop.shape[0] % num_sl
    dcm_vol_crop = dcm_vol_crop[mod_sl:]
    nslices = dcm_vol_crop.shape[0] // num_sl
    nx, ny = dcm_vol_crop.shape[1:]

    return np.mean(dcm_vol_crop.reshape(nslices, num_sl, nx, ny), axis=1)

def generate_new(args):
    src_dict = json.load(open(args.fpath_src_json, 'r'))
    base_path = src_dict['base_path']
    case_list = src_dict['cases']

    for case in case_list:
        try:
            print('Processing {}...'.format(case))
            dicom_dirs = suio.get_dicom_dirs('{}/{}'.format(base_path, case))
            study_uid = generate_uuid()

            op_cnum = case_num_transform(case)

            study_dir = '{}/{}'.format(args.output_path, op_cnum)
            if not os.path.isdir(study_dir):
                os.makedirs(study_dir)

            for fpath_dcm in dicom_dirs:
                ser_name = fpath_dcm.split('/')[-1]
                dcm_2d = slice_avg_3d_vol(fpath_dcm)
                dirpath_output = '{}/{}'.format(study_dir, ser_name)

                suio.write_dicoms(
                    fpath_dcm, dcm_2d, dirpath_output, series_desc_pre='', series_desc_post='2DSIM'
                )
                rename_dcm_meta(dirpath_output, study_uid)

        except Exception as exc:
            print('Error in processing {} - {}'.format(case, exc))
            continue

def get_aug_params(cases, base_params, ser_split, choice_params, shuffle=True):
    key_map = {
        "p": "pre", "l": "low", "f": "full"
    }
    param_list = []

    for k, v in ser_split.items():
        pcopy = base_params.copy()
        for ksplit in list(k):
            pcopy['augment_{}'.format(key_map[ksplit])] = True
        param_list.extend([pcopy] * v)

    param_list_new = []
    for idx, pdict in enumerate(param_list):
        dcopy = pdict.copy()
        dcopy['case_num'] = cases[idx]
        for k, v in choice_params.items():
            if not v:
                dcopy[k] = None
            else:
                dcopy[k] = np.random.choice(v)

        param_list_new.append(dcopy)

    if shuffle:
        np.random.shuffle(param_list_new)

    return param_list_new

def perform_augmentation(
    base_path, outpath, ring, motion, case_num, augment_pre, augment_low, augment_full,
    ring_lines=0, motion_deg=0, motion_trans=0, motion_num_tfm=0
):
    dicom_dirs = suio.get_dicom_dirs('{}/{}'.format(base_path, case_num))
    fpath_pre, fpath_low, fpath_full = dicom_dirs

    fpath_proc_list = []
    desc_suffix = ''
    study_uid = generate_uuid()
    if augment_pre:
        fpath_proc_list.append(fpath_pre)
        desc_suffix += 'P'
    if augment_low:
        fpath_proc_list.append(fpath_low)
        desc_suffix += 'L'
    if augment_full:
        fpath_proc_list.append(fpath_full)
        desc_suffix += 'F'

    for fpath_dcm in fpath_proc_list:
        dcm_2d, hdr = suio.dicom_files(fpath_dcm)
        ser_name = fpath_dcm.split('/')[-1]
        ser_suffix = desc_suffix
        if ring:
            ser_suffix += '_RNG'
            add_aliasing = AliasingLayer(pct_lines_to_corrupt=ring_lines, prob=1)
            dcm_ring, _ = add_aliasing.call([dcm_2d], None)
            dcm_2d = dcm_ring[0]

        if motion:
            ser_suffix += '_MOT'
            add_motion = tio.RandomMotion(
                degrees=motion_deg, translation=motion_trans, num_transforms=int(motion_num_tfm), image_interpolation='bspline'
            )
            dcm_motion = add_motion(dcm_2d[None, ...])
            dcm_2d = dcm_motion[0]

        cnum_mod = '{}_{}'.format(case_num, ser_suffix)
        outpath_mod = outpath.replace(case_num, cnum_mod)
        dirpath_ser_out = '{}/{}'.format(outpath_mod, ser_name)
        if not os.path.exists(dirpath_ser_out):
            os.makedirs(dirpath_ser_out)

        suio.write_dicoms(
            fpath_dcm, dcm_2d, dirpath_ser_out, series_desc_pre='', series_desc_post=ser_suffix,
            series_num=hdr.SeriesNumber
        )
        rename_dcm_meta(dirpath_ser_out, study_uid)

    for dcm_dir in dicom_dirs:
        if dcm_dir in fpath_proc_list: continue
        dest_dir = '{}/{}'.format(outpath_mod, dcm_dir.split('/')[-1])
        shutil.copytree(dcm_dir, dest_dir)
        rename_dcm_meta(dest_dir, study_uid)

def get_stat_dict(ring_mode, motion_mode, pdict):
    return {
        'Case': pdict['case_num'],
        'Pre Augmented': pdict['augment_pre'],
        'Low Augmented': pdict['augment_low'],
        'Full Augmented': pdict['augment_full'],
        'Ringing Artifacts': ring_mode,
        'Ringing Lines Affected': pdict['ring_lines'] or 0,
        'Motion Artifacts': motion_mode,
        'Motion Degree': pdict['motion_deg'] or 0,
        'Motion Translation': pdict['motion_trans'] or 0,
        'Motion Num Transforms': pdict['motion_num_tfm'] or 0
    }

def artifacts_augment(args):
    param_dict = json.load(open(args.fpath_src_json, 'r'))
    base_params = {
        'augment_pre': False,
        'augment_low': False,
        'augment_full': False
    }
    choice_params = {
        'ring_lines': [],
        'motion_deg': [],
        'motion_trans': [],
        'motion_num_tfm': []
    }

    if args.augment_mode in ['ring', 'both']:
        choice_params['ring_lines'] = param_dict['ring_lines']

    if args.augment_mode in ['motion', 'both']:
        choice_params['motion_deg'] = param_dict['motion_deg']
        choice_params['motion_trans'] = param_dict['motion_trans']
        choice_params['motion_num_tfm'] = param_dict['motion_num_tfm']

    param_list = get_aug_params(
        param_dict['cases'], base_params, ser_split=param_dict['ser_split'],
        choice_params=choice_params
    )

    stat_dicts = []
    for pdict in param_list:
        try:
            print('Processing {}...'.format(pdict['case_num']))
            dirpath_out = '{}/{}'.format(args.output_path, pdict['case_num'])
            ring_mode = (args.augment_mode in ['ring', 'both'])
            motion_mode = (args.augment_mode in ['motion', 'both'])
            perform_augmentation(
                param_dict['base_path'], outpath=dirpath_out, ring=ring_mode, motion=motion_mode,
                **pdict
            )

            stat_dicts.append(get_stat_dict(ring_mode, motion_mode, pdict))
        except Exception as exc:
            print('Error processing {} - {}'.format(pdict['case_num'], exc))
            continue

    pd.DataFrame(stat_dicts).to_csv('{}/sim_stats.csv'.format(args.output_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        usage=usage_str, description=description_str,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--fpath_src_json', action='store', dest='fpath_src_json', type=str,
    help='File path for source JSON which has the base path and list of case numbers to process',
    default='')
    parser.add_argument('--process_mode', action='store', dest='process_mode', type=str, help='Mode of processing - "generate_new" or "artifacts_augment". "generate_new" generates clean 2D datasets from the source 3D datasets. "artifacts_augment" takes clean 2D datasets and augments with random motion and ringing artifacts', default='generate_new')
    parser.add_argument('--augment_mode', action='store', dest='augment_mode', type=str, help='This argument tells whether augmentation is "ring", "motion" or "both"', default=None)
    parser.add_argument('--slice_thk', action='store', dest='slice_thk', type=int, help='Desired slice thickness for the 2D datasets', default=5)
    parser.add_argument('--skip_idx', action='store', dest='skip_idx', type=int, help='Number of slices to skip from the ends of the 3D volume', default=30)
    parser.add_argument('--output_path', action='store', dest='output_path', type=str, help='Directory path to write the DICOM files', default='')

    args = parser.parse_args()

    if args.process_mode == 'generate_new':
        generate_new(args)
    elif args.process_mode == 'artifacts_augment':
        artifacts_augment(args)
    else:
        raise ValueError('{} is not a valid process mode'.format(args.process_mode))
