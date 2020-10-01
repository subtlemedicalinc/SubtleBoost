import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import tempfile
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from HD_BET.run import run_hd_bet
from subtle.subtle_preprocess import dcm_to_sitk

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (12, 10)

def find_pre_contrast_series(dirpath_case):
    return sorted(
        [ser for ser in glob('{}/*BRAVO*'.format(dirpath_case))],
        key=lambda ser: int(ser.split('/')[-1].split('_')[0])
    )[0]

def get_plot_image(data, mask):
    img_stack = []
    data_tr = data.transpose(1, 0, 2, 3)

    t1_zero, t1_low, t1_full = data_tr[:3]
    sl = t1_zero.shape[0] // 2
    mask_sl = mask[sl]
    img_stack.append(t1_zero[sl] * mask_sl)
    img_stack.append(t1_low[sl] * mask_sl)
    img_stack.append(t1_full[sl] * mask_sl)

    if data.shape[1] == 4:
        t2 = data_tr[-1]
        img_stack.append(t2[sl] * mask_sl)

    return np.hstack(img_stack)

if __name__ == '__main__':
    pp_base_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data'
    dcm_path = pp_base_path.replace('preprocess/', '')
    save_path = os.path.join(pp_base_path, 'hdbet_masks')
    plot_path = os.path.join(save_path, 'plots')

    cases = sorted([
        f.split('/')[-1].replace('.npy', '')
        for f in glob('{}/*.npy'.format(pp_base_path))
    ])

    processed_cases = [
        f.split('/')[-1].replace('.npy', '')
        for f in glob('{}/*.npy'.format(save_path))
    ]

    cases = [c for c in cases if c not in processed_cases]

    for case_num in tqdm(cases, total=len(cases)):
        try:
            dirpath_precon = find_pre_contrast_series(os.path.join(dcm_path, case_num))
            ref_dcm = dcm_to_sitk(dirpath_precon)
            full_data = np.load(os.path.join(pp_base_path, '{}.npy'.format(case_num)))
            data_arr = full_data[0, :, 0]
            data_sitk = sitk.GetImageFromArray(data_arr)
            data_sitk.CopyInformation(ref_dcm)
            mask = None
            with tempfile.TemporaryDirectory() as tmpdir:
                fpath_input = '{}/input.nii.gz'.format(tmpdir)
                fpath_output = '{}/output.nii.gz'.format(tmpdir)
                fpath_mask = '{}/output_mask.nii.gz'.format(tmpdir)

                sitk.WriteImage(data_sitk, fpath_input)
                run_hd_bet(fpath_input, fpath_output, mode='fast', device=4, do_tta=False)
                mask = nib.load(fpath_mask).get_data().transpose(2, 1, 0)

            np.save(os.path.join(save_path, '{}.npy'.format(case_num)), mask)
            fpath_plot = os.path.join(plot_path, '{}.png'.format(case_num))
            plt_img = get_plot_image(full_data[0], mask)

            plt.imshow(plt_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(fpath_plot)
        except Exception as exc:
            print('ERROR in {}: {}'.format(case_num, exc))
