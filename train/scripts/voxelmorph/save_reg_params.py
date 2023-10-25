import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt
import subtle.utils.io as suio
from subtle.subtle_preprocess import register_im, dcm_to_sitk
from subtle.utils.experiment import get_experiment_data
from scipy.ndimage import affine_transform
import nibabel as nib
from glob import glob
from tqdm import tqdm

plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (12, 10)

def preprocess_vol(vol):
    vol = vol / vol.mean()
    vol = np.interp(vol, (vol.min(), vol.max()), (0, 1))
    return vol

def get_sitk_affn_mat(reg_params):
    tr_params = np.array(reg_params[0]['TransformParameters']).astype(np.float32)
    tr_params = np.array(np.array_split(tr_params, 4)).T
    tr_4x4 = np.append(tr_params, [[0, 0, 0, 1]], axis=0)
    tr_4x4_inv = np.linalg.inv(tr_4x4)
    return tr_4x4_inv[:-1]

if __name__ == '__main__':
    cases = get_experiment_data('stanford_sri',
                            dirpath_exp='/home/srivathsa/projects/SubtleGad/train/configs/experiments/',
                            dataset='all',
                            fname='data_vmorph'
                           )

    src_path = '/home/srivathsa/projects/studies/gad/stanford/data'
    dest_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/aff_params'

    for cnum in tqdm(cases, total=len(cases)):
        fpath_pre, fpath_low, _ = suio.get_dicom_dirs('{}/{}'.format(src_path, cnum))

        vol1, _ = suio.dicom_files(fpath_pre)
        vol1 = preprocess_vol(vol1)

        vol2, _ = suio.dicom_files(fpath_low)
        vol2 = preprocess_vol(vol2)

        ref_zero = dcm_to_sitk(fpath_pre)
        ref_low = dcm_to_sitk(fpath_low)

        pmap = sitk.GetDefaultParameterMap('affine')
        vol2_reg, reg_params = register_im(vol1, vol2, param_map=pmap, verbose=0,
                               ref_fixed=ref_zero, ref_moving=ref_low)

        aff_mtx = get_sitk_affn_mat(reg_params)
        np.save('{}/{}.npy'.format(dest_path, cnum), aff_mtx)
