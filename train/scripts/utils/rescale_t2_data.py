import os
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.morphology import binary_closing

from subtle.subtle_preprocess import scale_im as hist_eq

if __name__ == '__main__':
    scale_constant = 1.389
    base_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data'
    data_files = sorted(glob('{}/*.npy'.format(base_path)))
    data_files = [f for f in data_files if '_sc' not in f]

    for fpath in tqdm(data_files, total=len(data_files)):
        try:
            data = np.load(fpath)

            t1_low = data[0, :, 1]
            t2_vol = data[0, :, 3]

            t2_vol  = hist_eq(t1_low, t2_vol) * scale_constant
            mask = data[1, :, 2] >= 0.1
            mask = binary_closing(mask)

            t2_vol_mask = t2_vol * mask

            data_new = data.copy()
            data_new[0, :, 3] = t2_vol
            data_new[1, :, 3] = t2_vol_mask

            fpath_new = fpath.replace('.npy', '_sc.npy')
            np.save(fpath_new, data_new)
            os.remove(fpath)
            print('Saving new volume to {}...'.format(fpath_new))
        except Exception as ex:
            print('Error in {} {}'.format(fpath.split('/')[-1], ex))
