import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import tempfile
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pydicom
from scipy.ndimage.interpolation import zoom as zoom_interp
import sigpy as sp

from HD_BET.run import run_hd_bet
from subtle.subtle_preprocess import dcm_to_sitk, center_crop, zero_pad
import subtle.utils.io as suio

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (12, 10)

def find_pre_contrast_series(dirpath_case):
    ser_nums = []
    for ser_dir in glob('{}/*'.format(dirpath_case)):
        if 'T2' in ser_dir: continue
        dcm_file = [
            f for f in glob('{}/**/*'.format(ser_dir), recursive=True)
            if os.path.isfile(f) and 'XX' not in f
        ][0]
        dcm = pydicom.dcmread(dcm_file)
        ser_nums.append((ser_dir, dcm.SeriesNumber))

    ser_nums = sorted(ser_nums, key=lambda s: int(s[1]))
    dpath_precon = ser_nums[0][0]
    fpath_dcm = [fp for fp in glob('{}/**/*.dcm'.format(dpath_precon), recursive=True)][0]
    return str(Path(fpath_dcm).parent.absolute())

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

def kw_not_in(s, kws):
    for kw in kws:
        if kw in s:
            return False
    return True

def reshape_input(vol, case_num, ref_sitk):
    if 'Id' in case_num:
        # crop from 256 -> 240 and then zoom by 0.75 to reach 320
        pixel_spacing = np.array(ref_sitk.GetSpacing())[::-1]
        ref_size = np.array(ref_sitk.GetSize())[::-1]
        rs_size = pixel_spacing * ref_size
        rs_size = [int(np.round(r)) for r in rs_size]

        vol = sp.util.resize(vol, rs_size)
        zf = np.array([1., 1., 1.]) / np.array(ref_sitk.GetSpacing())[::-1]
        vol = zoom_interp(vol, zf)
    elif 'Prisma' in case_num:
        # crop from 256,256 -> 256,232
        vol = sp.util.resize(vol, (vol.shape[0], 256, 232))
    elif 'Patient' in case_num:
        # zoom by 2 from 256 -> 512
        vol = zoom_interp(vol, (1, 2, 2))
    return vol

def reshape_mask(vol, case_num, ref_sitk):
    # reverse the process in reshape_input
    if 'Id' in case_num:
        zf = np.array(ref_sitk.GetSpacing())[::-1]/ np.array([1., 1., 1.])
        vol = zoom_interp(vol, zf)
        vol = sp.util.resize(vol, (vol.shape[0], 256, 256))
    elif 'Prisma' in case_num:
        vol = sp.util.resize(vol, (vol.shape[0], 256, 256))
    elif 'Patient' in case_num:
        vol = zoom_interp(vol, (1, 0.5, 0.5))
    return vol

if __name__ == '__main__':
    pp_base_path = '/home/srivathsa/projects/studies/gad/gen_siemens/preprocess/data'
    # dcm_path = pp_base_path.replace('preprocess/', '')
    dcm_path = '/home/srivathsa/projects/studies/gad/gen_siemens/data'
    # save_path = os.path.join(pp_base_path, 'hdbet_masks')
    save_path = '/home/srivathsa/projects/studies/gad/gen_siemens/preprocess/data/hdbet_masks'
    # plot_path = os.path.join(save_path, 'hdbet_plots')
    plot_path = '/home/srivathsa/projects/studies/gad/gen_siemens/preprocess/data/hdbet_masks/plots'

    cases = sorted([
        f.split('/')[-1].replace('.h5', '')
        for f in glob('{}/*.h5'.format(pp_base_path))
        if kw_not_in(f, ['meta', 'TwoDim'])
    ])

    processed_cases = [
        f.split('/')[-1].replace('.npy', '')
        for f in glob('{}/*.npy'.format(save_path))
    ]

    cases = [c for c in cases if c not in processed_cases]
    print('cases', cases)

    for case_num in tqdm(cases, total=len(cases)):
        try:
            dirpath_precon = find_pre_contrast_series(os.path.join(dcm_path, case_num))

            '''Only for bch'''
            # dirs_mprage = [
            #     d for d in glob('{}/*'.format(os.path.join(dcm_path, case_num)))
            #     if 'mprage' in d.lower()
            # ]
            #
            # dirs_smr = [
            #     d for d in glob('{}/*'.format(os.path.join(dcm_path, case_num)))
            #     if 'smr' in d.lower()
            # ]
            #
            # dirpath_precon = dirs_mprage[0] if len(dirs_mprage) == 1 else dirs_smr[0]
            '''Only for bch'''

            ref_dcm = dcm_to_sitk(dirpath_precon)
            ref_dims = ref_dcm.GetSize()[::-1]
            # full_data = np.load(os.path.join(pp_base_path, '{}.npy'.format(case_num)))
            full_data = suio.load_file(os.path.join(pp_base_path, '{}.h5'.format(case_num)))
            data_arr = full_data[:, 0]
            data_arr = reshape_input(data_arr, case_num, ref_dcm)

            data_sitk = sitk.GetImageFromArray(data_arr)
            data_sitk.CopyInformation(ref_dcm)
            mask = None
            with tempfile.TemporaryDirectory() as tmpdir:
                fpath_input = '{}/input.nii.gz'.format(tmpdir)
                fpath_output = '{}/output.nii.gz'.format(tmpdir)
                fpath_mask = '{}/output_mask.nii.gz'.format(tmpdir)

                sitk.WriteImage(data_sitk, fpath_input)
                run_hd_bet(fpath_input, fpath_output, mode='fast', device=2, do_tta=False)
                mask = nib.load(fpath_mask).get_data().transpose(2, 1, 0)
                mask = reshape_mask(mask, case_num, ref_dcm)

            np.save(os.path.join(save_path, '{}.npy'.format(case_num)), mask)
            fpath_plot = os.path.join(plot_path, '{}.png'.format(case_num))
            plt_img = get_plot_image(full_data, mask)

            plt.imshow(plt_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(fpath_plot)
        except Exception as exc:
            print('ERROR in {}: {}'.format(case_num, exc))
