import numpy as np
from glob import glob
import os
from tqdm import tqdm

base_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data'
dirpath_src = os.path.join(base_path, 'old_mask')
dirpath_dest = base_path
dirpath_masks = os.path.join(base_path, 'hdbet_masks')

if __name__ == '__main__':
    cases = sorted([
        c.split('/')[-1].replace('.npy', '')
        for c in glob('{}/*.npy'.format(dirpath_src))
    ])

    ignore_cases = [
        c.split('/')[-1].replace('.npy', '')
        for c in glob('{}/*.npy'.format(dirpath_dest))
    ]

    cases = [c for c in cases if c not in ignore_cases]

    for case_num in tqdm(cases, total=len(cases)):
        try:
            fpath_hdbet_mask = os.path.join(dirpath_masks, '{}.npy'.format(case_num))
            mask = np.load(fpath_hdbet_mask)
            full_data = np.load(os.path.join(dirpath_src, '{}.npy'.format(case_num)))
            mask_full = np.repeat(mask[:, np.newaxis, :, :], full_data.shape[2], axis=1)

            data_mask_new = mask_full * full_data[0]
            full_data[1] = data_mask_new

            np.save(os.path.join(dirpath_dest, '{}.npy'.format(case_num)), full_data)
        except Exception as exc:
            print('ERROR in {}: {}'.format(case_num, exc))
