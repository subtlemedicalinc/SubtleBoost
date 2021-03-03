from glob import glob
from tqdm import tqdm
import numpy as np
from subtle.utils.io import save_data_h5
from subtle.utils.experiment import get_experiment_data

if __name__ == '__main__':
    base_path = '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data'
    out_path = '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data'

    cases = get_experiment_data('tiantan_t2', dataset='train')

    for case_num in cases:
        try:
            print('Processing {}...'.format(case_num))

            fpath_npy = '{}/{}.npy'.format(base_path, case_num)
            data_all = np.load(fpath_npy)

            data = data_all[0, :, :3, ...]
            data_mask = data_all[1, :, :3, ...]

            save_data_h5('{}/{}.h5'.format(out_path, case_num), data=data, data_mask=data_mask)
        except Exception as exc:
            print('ERROR in {}: {}'.format(case_num, exc))
