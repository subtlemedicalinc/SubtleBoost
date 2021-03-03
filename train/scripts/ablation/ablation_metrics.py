import os
from glob import glob
import argparse
import warnings
import json
import tempfile

from tqdm import tqdm
from skimage.measure import compare_ssim
import numpy as np
import pydicom
import SimpleITK as sitk
import h5py
import shutil
from scipy.ndimage.interpolation import zoom as zoom_interp

def get_dicom_vol(dirpath_dicom):
    dcm_files = sorted([f for f in glob('{}/**/*.dcm'.format(dirpath_dicom), recursive=True)])
    return np.array([pydicom.dcmread(f).pixel_array for f in dcm_files])

def nrmse(x_truth, x_predict, axis=None):
    return np.linalg.norm(x_truth - x_predict, axis=axis) / np.linalg.norm(x_truth, axis=axis)

def normalize_ims(x_truth, x_predict):
    if np.all(x_truth >= 0):
        max_val = np.max(x_truth)
        x_truth_nrm = x_truth / max_val
        x_predict_nrm = x_predict / max_val
    else:
        max_val = np.max(abs(x_truth))
        x_truth_nrm = x_truth / max_val / 2.
        x_predict_nrm = x_predict / max_val / 2.
    return x_truth_nrm, x_predict_nrm

def psnr(x_truth, x_predict, axis=None, dynamic_range=None):
    if dynamic_range is None:
        x_truth, x_predict = normalize_ims(x_truth, x_predict)
        dynamic_range = 1.

    if axis is None:
        nrm = len(x_truth.ravel())
    else:
        nrm = x_truth.shape[axis]

    MSE = np.linalg.norm(x_truth - x_predict, axis=axis)**2 / nrm

    return 20 * np.log10(dynamic_range) - 10 * np.log10(MSE)

def ssim(x_truth, x_predict, axis=None, dynamic_range=None):
    if dynamic_range is None:
        x_truth, x_predict = normalize_ims(x_truth, x_predict)
        dynamic_range = 1.

    if axis is None:
        nrm = len(x_truth.ravel())
    else:
        nrm = x_truth.shape[axis]

    if x_truth.dtype != x_predict.dtype:
        warnings.warn('x_truth.dtype == {} != {} == x_predict.dtype. Casting x_predict to x_truth'.format(x_truth.dtype, x_predict.dtype))
        x_predict = x_predict.astype(dtype=x_truth.dtype)

    return compare_ssim(x_truth, x_predict, data_range=dynamic_range)

def rename_series(dirpath_series, desc_suffix, outpath, cs=False):
    dcm_files = [(pydicom.dcmread(f), f) for f in glob('{}/*.dcm'.format(dirpath_series))]
    if cs:
        desc_suffix = '_cs'

    print('Renaming {} with "{}" suffix'.format(dirpath_series, desc_suffix))

    series_desc = '{}_{}'.format(dcm_files[0][0].SeriesDescription, desc_suffix)

    for dcm_file in tqdm(dcm_files, total=len(dcm_files)):
        dcm_file[0].SeriesDescription = series_desc

        fpath_out = os.path.join(outpath, dcm_file[1].split('/')[-1])
        dcm_file[0].save_as(fpath_out)

def process_renaming(dirpath_study, rename_dict={}, cs=False):
    dict_snum = {}
    keys = ['zero', 'low', 'full']

    dirpaths_series = [d for d in glob('{}/*'.format(dirpath_study))]

    for series_path in dirpaths_series:
        dcm_files = [f for f in glob('{}/*.dcm'.format(series_path))]
        dcm = pydicom.dcmread(dcm_files[0])
        dict_snum[int(dcm.SeriesNumber)] = series_path

    dict_suffixes = {}
    for i, snum in enumerate(sorted(dict_snum.keys())):
        dict_suffixes[rename_dict[keys[i]]] = dict_snum[snum]

    for k, v in dict_suffixes.items():
        rename_series(v, k, cs)

def get_series_in_order(input_dir):
    series = glob('{}/*'.format(input_dir))
    series_num = sorted([
        (d, int(pydicom.dcmread(glob('{}/**/*.dcm'.format(d), recursive=True)[0]).SeriesNumber))
        for d in series
    ], key=lambda d: d[1])
    return [s[0] for s in series_num]

def scale_im(im_fixed, im_moving, levels=1024, points=7, mean_intensity=True):
    sim0 = sitk.GetImageFromArray(im_fixed.squeeze())
    sim1 = sitk.GetImageFromArray(im_moving.squeeze())

    hm = sitk.HistogramMatchingImageFilter()
    hm.SetNumberOfHistogramLevels(levels)
    hm.SetNumberOfMatchPoints(points)

    if mean_intensity:
        hm.ThresholdAtMeanIntensityOn()
    sim_out = hm.Execute(sim1, sim0)
    im_out = sitk.GetArrayFromImage(sim_out)
    return im_out

def load_h5_metadata(h5_file, key='metadata'):
    metadata = {}
    with h5py.File(h5_file, 'r') as F:
        for k in F[key].keys():
            metadata[k] = np.array(F[key][k])
    return metadata

def _get_crop_range(shape_num):
    if shape_num % 2 == 0:
        start = end = shape_num // 2
    else:
        start = (shape_num + 1) // 2
        end = (shape_num - 1) // 2

    return (start, end)

def center_crop(img, ref_img):
    s = []
    e = []

    for i, sh in enumerate(img.shape):
        if sh > ref_img.shape[i]:
            diff = sh - ref_img.shape[i]
            if diff == 1:
                s.append(0)
                e.append(sh-1)
            else:
                start, end = _get_crop_range(diff)
                s.append(start)
                e.append(-end)
        else:
            s.append(0)
            e.append(sh)

    new_img = img[s[0]:e[0], s[1]:e[1], s[2]:e[2]]

    if new_img.shape[1] != new_img.shape[2]:
        new_img = zoom_interp(new_img, [1, ref_img.shape[1]/new_img.shape[1], ref_img.shape[2]/new_img.shape[2]])

    return new_img

app_base = '/home/srivathsa/projects/SubtleGad/app'
config_path = 'app_config.yml'
license_file = '/home/srivathsa/license_gad.json'

rename_dicts = {
    '1_ablation_2d': {
        'zero': 'zd_abl_2d',
        'low': 'ld_abl_2d',
        'full': 'fd_abl_2d'
    },
    '2_ablation_7ch': {
        'zero': 'zd_abl_7ch',
        'low': 'ld_abl_7ch',
        'full': 'fd_abl_7ch'
    },
    '3_ablation_mpr': {
        'zero': 'zd_abl_mpr',
        'low': 'ld_abl_mpr',
        'full': 'fd_abl_mpr'
    },
    '4_ablation_vgg': {
        'zero': 'zd_abl_vgg',
        'low': 'ld_abl_vgg',
        'full': 'fd_abl_vgg'
    },
    '5_ablation_enh': {
        'zero': 'zd_abl_enh',
        'low': 'ld_abl_enh',
        'full': 'fd_abl_enh'
    }
}

if __name__ == '__main__':
    usage_str = 'usage: %(prog)s [options]'
    description_str = 'compute performance stats for ablation study'

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', action='store', dest='input', type=str,  help='DICOM directory path of the case for which metrics are to be computed', default=None)
    parser.add_argument('--pp_base', action='store', dest='pp_base', type=str, help='Base path for preprocessing file', default=None)
    parser.add_argument('--output', action='store', dest='output', type=str, help='Output path for processed DICOMs and metrics JSON', default=None)

    args = parser.parse_args()
    case_num = args.input.split('/')[-1]

    zero_dir, low_dir, _ = get_series_in_order(args.input)

    fpath_pp = os.path.join(args.pp_base, case_num)
    if os.path.exists('{}.h5'.format(fpath_pp)):
        h5_f = h5py.File('{}.h5'.format(fpath_pp), 'r')
        gt_vol = np.array(h5_f['data'])[:, 2]
    else:
        gt_vol = np.load('{}.npy'.format(fpath_pp))[0, :, 2]

    meta = load_h5_metadata('{}/{}_meta.h5'.format(args.pp_base, case_num))
    sc = meta['scale_global'][0][0]

    if 'original_size' in meta:
        if meta['original_size'][0] != gt_vol.shape[1] or meta['original_size'][1] != gt_vol.shape[2]:
            gt_vol = center_crop(
                gt_vol, np.zeros(
                    (gt_vol.shape[0], meta['original_size'][0], meta['original_size'][1])
                )
            )

    metrics_dict = {}

    with tempfile.TemporaryDirectory() as temp_path:
        for ablation_key in sorted(rename_dicts.keys()):
            abl_base_path = os.path.join(temp_path, case_num, ablation_key)
            rename_obj = rename_dicts[ablation_key]
            dirpath_zero = os.path.join(abl_base_path, 'zero')
            dirpath_low = os.path.join(abl_base_path, 'low')

            if not os.path.isdir(abl_base_path):
                os.makedirs(abl_base_path)

            if not os.path.isdir(dirpath_zero):
                os.makedirs(dirpath_zero)

            if not os.path.isdir(dirpath_low):
                os.makedirs(dirpath_low)

            out_base_path = os.path.join(args.output, case_num, ablation_key)
            if os.path.isdir(out_base_path):
                shutil.rmtree(out_base_path)
            os.makedirs(out_base_path)

            rename_series(zero_dir, desc_suffix=rename_obj['zero'], outpath=dirpath_zero)
            rename_series(low_dir, desc_suffix=rename_obj['low'], outpath=dirpath_low)

            app_cmd = 'python {app_base}/infer.py {input_dir} {output_dir} -c {config_file} -l {license_file}'.format(
                app_base=app_base, input_dir=abl_base_path, output_dir=out_base_path,
                config_file=config_path, license_file=license_file
            )

            print('Running the following command:')
            print('-----')
            print(app_cmd)
            print('-----')
            os.system(app_cmd)

            pred_vol = get_dicom_vol(out_base_path)
            pred_vol = pred_vol / sc

            sl = pred_vol.shape[0] // 2

            metrics_dict[ablation_key] = {
                'psnr_vol': str(psnr(gt_vol, pred_vol)),
                'psnr_sl': str(psnr(gt_vol[sl], pred_vol[sl])),
                'ssim_vol': str(ssim(gt_vol, pred_vol)),
                'ssim_sl': str(ssim(gt_vol[sl], pred_vol[sl]))
            }

    fpath_json = os.path.join(args.output, case_num, 'metrics.json')
    with open(fpath_json, 'w') as f:
        json.dump(metrics_dict, f)
    print('Metrics for {} saved successfully to {}'.format(case_num, fpath_json))
