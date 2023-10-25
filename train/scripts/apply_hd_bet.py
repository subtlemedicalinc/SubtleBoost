import numpy as np
from glob import glob
import os
from tqdm import tqdm
import subtle.utils.io as suio
from multiprocessing import Pool

base_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data'
dirpath_src = os.path.join(base_path, 'old_mask')
dirpath_dest = base_path
dirpath_masks = os.path.join(base_path, 'hdbet_masks')

def kw_not_in(s, kws):
    for kw in kws:
        if kw in s:
            return False
    return True

def process_single_case(params):
    print('Processing {}...'.format(params['case_num']))
    try:
        dirpath_masks = params['dirpath_masks']
        dirpath_src = params['dirpath_src']
        case_num = params['case_num']

        fpath_hdbet_mask = os.path.join(dirpath_masks, '{}.npy'.format(case_num))
        mask = np.load(fpath_hdbet_mask)
        full_data = suio.load_file(os.path.join(dirpath_src, '{}.npy'.format(case_num)), params={'h5_key': 'all'})
        vol_shape = np.array([full_data.shape[1], full_data.shape[3], full_data.shape[4]])

        mask_full = np.repeat(mask[:, np.newaxis, :, :], full_data.shape[2], axis=1)

        data_mask_new = mask_full * full_data[0]
        full_data[1] = data_mask_new

        np.save(os.path.join(dirpath_dest, '{}_full.npy'.format(case_num)), full_data)
    except Exception as exc:
        print('ERROR in {}: {}'.format(case_num, exc))

if __name__ == '__main__':
    cases = sorted([
        c.split('/')[-1].replace('.h5', '')
        for c in glob('{}/*.h5'.format(dirpath_src))
    ])

    ignore_cases = [
        c.split('/')[-1].replace('.h5', '')
        for c in glob('{}/*.h5'.format(dirpath_dest))
        if kw_not_in(c, ['meta', 'TwoDim'])
    ]

    cases = [c for c in cases if c not in ignore_cases]

    cases = sorted([
        c for c in open('scripts/hd_bet_cases.txt', 'r').read().split('\n')
        if len(c) > 0
    ])

    proc_params = []

    for case_num in tqdm(cases, total=len(cases)):
        proc_params.append({
            'dirpath_masks': dirpath_masks,
            'dirpath_src': dirpath_src,
            'case_num': case_num
        })

    process_pool = Pool(processes=16, initializer=None, initargs=None)
    _ = process_pool.map(process_single_case, proc_params)
    process_pool.close()
    process_pool.join()
