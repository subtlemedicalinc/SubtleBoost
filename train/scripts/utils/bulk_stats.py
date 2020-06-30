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
    parser.add_argument('--file_ext', action='store', dest='file_ext', type=str, help='npy or h5', default='npy')
    parser.add_argument('--resize', action='store_true', dest='resize', help='center crop', default=False)

    args = parser.parse_args()

    base_path = args.base_path
    pp_path = args.pp_path
    csv_path = args.csv_path
    file_ext = args.file_ext

    # case_list = sorted([d.split('/')[-1] for d in glob('{}/Patient*'.format(base_path))])
    # case_list = ['NO{}'.format(n) for n in range(25, 84)]
    case_list = ['NO71', 'NO75', 'NO78', 'NO79', 'NO80', 'NO81', 'NO82', 'NO83']
    # case_list = ["Patient_0187", "Patient_0204", "Patient_0207", "Patient_0213", "Patient_0215", "Patient_0267", "Patient_0351", "Patient_0378", "Patient_0409", "Patient_0435", "Patient_0470", "Patient_0492", "Patient_0495", "Patient_0572"]
    # case_list = ['101_Id_007', '101_Id_033', '101_Id_052', '101_Id_061', '101_Id_066', 'Id0029', 'Id0032']
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

            if args.resize:
                # gt = center_crop(gt, pred)
                # gt = center_crop(gt, np.zeros((pred.shape[0], 240, 240)))
                gt = zoom_interp(gt, [1, 2, 2])

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
