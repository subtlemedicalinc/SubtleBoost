import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import tempfile
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pydicom

from HD_BET.run import run_hd_bet
from subtle.subtle_preprocess import dcm_to_sitk, center_crop, zero_pad
import subtle.utils.io as suio

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (12, 10)

def get_plot_image(data, mask):
    sl = data.shape[0] // 2
    img1 = data[sl]
    img2 = data[sl] * mask[sl]

    return np.hstack([img1, img2])

if __name__ == '__main__':
    base_path = '/home/srivathsa/projects/studies/gad/smr_bet/nifti'
    save_path = '/home/srivathsa/projects/studies/gad/smr_bet/masks'
    plot_path = '/home/srivathsa/projects/studies/gad/smr_bet/plots'

    cases = sorted([d.split('/')[-1] for d in glob('{}/*'.format(base_path))])
    proc_cases = sorted([d.split('/')[-1] for d in glob('{}/*'.format(plot_path))])

    cases = [c for c in cases if c not in proc_cases]
    cases = ['fish-kilo-sad-massachusetts']

    for case_num in tqdm(cases, total=len(cases)):
        try:
            dpath_case = os.path.join(base_path, case_num)
            dpath_plot = os.path.join(plot_path, case_num)
            os.makedirs(dpath_plot, exist_ok=True)
            for fpath_nii in glob('{}/*.nii.gz'.format(dpath_case)):
                fpath_out = fpath_nii.replace('.nii.gz', '_tmp.nii.gz')
                run_hd_bet(fpath_nii, fpath_out, mode='fast', device=1, do_tta=False)
                fpath_mask = fpath_out.replace('_tmp', '_mask')
                os.rename(fpath_out.replace('_tmp', '_tmp_mask'), fpath_mask)
                # os.remove(fpath_out)

                full_data = nib.load(fpath_nii).get_data()
                full_data = full_data / full_data.mean()
                full_data = np.interp(full_data, (full_data.min(), full_data.max()), (0, 1))
                mask = nib.load(fpath_mask).get_data()

                plt_img = get_plot_image(full_data, mask)
                fpath_plot = os.path.join(plot_path, case_num, '{}.png'.format(fpath_nii.split('/')[-1].replace('.nii.gz', '')))

                plt_img = get_plot_image(full_data, mask)
                plt.imshow(plt_img)
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(fpath_plot)
        except Exception as e:
            print('ERROR in {}: {}'.format(case_num, e))
