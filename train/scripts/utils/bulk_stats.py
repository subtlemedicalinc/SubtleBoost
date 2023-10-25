import os
import pydicom
from glob import glob
import numpy as np
import subtle.subtle_metrics as sumetrics
import subtle.utils.io as suio
from subtle.subtle_preprocess import center_crop
from scipy.ndimage.interpolation import zoom as zoom_interp
import pandas as pd
import h5py
from tqdm import tqdm
import argparse

def get_dicom_vol(dirpath_dicom):
    dcm_files = sorted([f for f in glob('{}/*.dcm'.format(dirpath_dicom))])
    return np.array([pydicom.dcmread(f).pixel_array for f in dcm_files])

usage_str = 'usage: %(prog)s [options]'
description_str = 'compute performance stats from data'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_path', action='store', dest='base_path', type=str,  help='inference base path', default=None)
    parser.add_argument('--pp_path', action='store', dest='pp_path', type=str, help='preprocess data path', default=None)
    parser.add_argument('--csv_path', action='store', dest='csv_path', type=str, help='ground truth file')
    parser.add_argument('--file_ext', action='store', dest='file_ext', type=str, help='npy or h5', default='h5')
    parser.add_argument('--resize', action='store_true', dest='resize', help='center crop', default=False)

    args = parser.parse_args()

    base_path = args.base_path
    pp_path = args.pp_path
    csv_path = args.csv_path
    file_ext = args.file_ext

    case_list = [
        "101_Id_051", "101_Id_066", "Id0032", "NO108", "NO113", "NO120", "NO129", "NO130", "NO18", "NO26",
        "NO54", "NO55", "NO56", "NO6", "NO60", "NO62", "NO67", "NO71", "NO79", "Patient_0087",
        "Patient_0090", "Patient_0134", "Patient_0157", "Patient_0172", "Patient_0173", "Patient_0178",
        "Patient_0255", "Patient_0269", "Patient_0276", "Patient_0286", "Patient_0333", "Patient_0342",
        "Patient_0353", "Patient_0375", "Patient_0400", "Patient_0408", "Patient_0486", "Patient_0526",
        "Patient_0535", "Patient_0538", "Patient_0556", "Patient_0575", "Prisma1", "Prisma21", "Prisma22",
        "Prisma23", "Prisma3", "Prisma4", "Prisma6", "Prisma9"
    ]
    stats_arr = []

    for case in tqdm(case_list, total=len(case_list)):
        try:
            pred = get_dicom_vol('{}/{}/{}_SubtleGad'.format(base_path, case, case))

            fpath_meta = '{}/{}_meta.h5'.format(pp_path, case)
            meta = suio.load_h5_metadata(fpath_meta)
            sc = meta['scale_global'][0][0]

            pred = pred / sc

            fpath_gt = '{}/{}.{}'.format(pp_path, case, file_ext)
            if file_ext == 'npy':
                gt = np.load(fpath_gt)[0, :, 2, ...]
            else:
                gt = suio.load_file(fpath_gt)[:, 2]

            if gt.shape[1] != pred.shape[1] or gt.shape[2] != pred.shape[2]:
                if 'Id' in case:
                    crop_to_size = meta['old_spacing_zero'] * np.array(pred.shape)
                    gt = center_crop(gt, np.zeros((gt.shape[0], int(crop_to_size[1]), int(crop_to_size[2]))))
                    resample_size = np.array([1, 1, 1]) / meta['old_spacing_zero']
                    gt = zoom_interp(gt, resample_size)
                elif 'Prisma' in case:
                    gt = center_crop(gt, np.zeros_like(pred))
                else:
                    gt = zoom_interp(gt, [1, 2, 2])

            # if args.resize:
            #     gt = center_crop(gt, pred)
            #     # gt = center_crop(gt, np.zeros((pred.shape[0], 240, 240)))
            #     # gt = zoom_interp(gt, [1, 2, 2])

            psnr = sumetrics.psnr(gt, pred)
            ssim = sumetrics.ssim(gt, pred)

            stats_arr.append({
                'case': case,
                'psnr': psnr,
                'ssim': ssim
            })
        except Exception as err:
            print('ERROR in {}: {}'.format(case, err))
            continue

    df_stats = pd.DataFrame(stats_arr)
    df_stats.to_csv(csv_path)
