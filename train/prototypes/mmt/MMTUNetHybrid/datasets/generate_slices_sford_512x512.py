import os
import shutil
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as mae

def preprocess_data(data, con_idxs=[0, 2, 3, 4]):
    # data = data[0]
    # data = data.transpose(1, 0, 2, 3)
    # con_idxs = [0, 1, 2, 3] if data.shape[0] == 4 else con_idxs
    data = data[con_idxs]
    for idx in np.arange(data.shape[0]):
        data[idx] = data[idx] / data[idx].mean()
        data[idx] = np.clip(data[idx], 0, data[idx].max())

    data = data.astype(np.float32)
    return data


if __name__ == '__main__':
    src_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data'
    ref_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data_mmt/full_brain_256'
    dest_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data_mmt/full_brain_512'
    fail_cases = []
    for split in ['train', 'test', 'val']:
        print('Processing {} cases...'.format(split))
        con_idxs = [0, 2, 3, 4]
        cases = sorted([f.split('/')[-1] for f in glob('{}/{}/*'.format(ref_path, split))])
        proc_cases = sorted([f.split('/')[-1] for f in glob('{}/{}/*'.format(dest_path, split))])
        cases = [c for c in cases if c not in proc_cases]

        for cnum in tqdm(cases, total=len(cases)):
            try:
                case_data = np.load('{}/{}.npy'.format(src_path, cnum))
                proc_data = preprocess_data(case_data, con_idxs)

                save_path = '{}/{}/{}'.format(dest_path, split, cnum)
                os.makedirs(save_path, exist_ok=True)

                for sl_idx in np.arange(proc_data.shape[1]):
                    fname = '{}/{:03d}.npy'.format(save_path, sl_idx)
                    np.save(fname, proc_data[:, sl_idx, ...])
            except Exception as err:
                print('ERROR in {}:{}'.format(cnum, err))
                print('data shape -', case_data.shape)
                fail_cases.append(cnum)
    print('Failed to process the below cases')
    print(fail_cases)
