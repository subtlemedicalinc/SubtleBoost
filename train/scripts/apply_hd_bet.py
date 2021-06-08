import numpy as np
from glob import glob
import os
from tqdm import tqdm
import subtle.utils.io as suio

base_path = '/home/srivathsa/projects/studies/gad/radnet/preprocess/data'
dirpath_src = os.path.join(base_path, 'old_mask')
dirpath_dest = base_path
dirpath_masks = os.path.join(base_path, 'hdbet_masks')

def kw_not_in(s, kws):
    for kw in kws:
        if kw in s:
            return False
    return True

if __name__ == '__main__':
    cases = sorted([
        c.split('/')[-1].replace('.h5', '')
        for c in glob('{}/*.h5'.format(dirpath_src))
    ])

    ignore_cases = [
        c.split('/')[-1].replace('.h5', '')
        for c in glob('{}/*.h5'.format(dirpath_dest))
        if kw_not_in(c, ['meta', 'TwoDim', 'Prisma'])
    ]

    cases = [c for c in cases if c not in ignore_cases]

    for case_num in tqdm(cases, total=len(cases)):
        try:
            fpath_hdbet_mask = os.path.join(dirpath_masks, '{}.npy'.format(case_num))
            mask = np.load(fpath_hdbet_mask)
            full_data = suio.load_file(os.path.join(dirpath_src, '{}.h5'.format(case_num)), params={'h5_key': 'all'})
            vol_shape = np.array([full_data.shape[1], full_data.shape[3], full_data.shape[4]])

            if mask.shape[0] > vol_shape[0]:
                mask = center_crop(mask, np.zeros(vol_shape))
            elif mask.shape[0] < vol_shape[0]:
                sdiff = vol_shape[0] - mask.shape[0]
                npad = []
                if sdiff == 1:
                    npad = [(1, 0), (0, 0), (0, 0)]
                else:
                    diff_part = sdiff // 2
                    diff_rem = sdiff - diff_part
                    npad = [(diff_part, diff_rem), (0, 0), (0, 0)]
                mask = np.pad(mask, pad_width=npad, mode='constant', constant_values=0)

            mask_full = np.repeat(mask[:, np.newaxis, :, :], full_data.shape[2], axis=1)

            data_mask_new = mask_full * full_data[0]
            full_data[1] = data_mask_new

            # np.save(os.path.join(dirpath_dest, '{}.npy'.format(case_num)), full_data)
            suio.save_data_h5(os.path.join(dirpath_dest, '{}.h5'.format(case_num)), data=full_data[0], data_mask=full_data[1])
        except Exception as exc:
            print('ERROR in {}: {}'.format(case_num, exc))
