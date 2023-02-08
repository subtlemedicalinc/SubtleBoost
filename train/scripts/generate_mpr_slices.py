import os
import numpy as np
from glob import glob
from skimage.transform import resize
import sigpy as sp
from tqdm import tqdm

def process_mpr(vol, plane):
    if plane == 'sag':
        tr = (2, 0, 1)
        x1 = 100
    else:
        tr = (1, 0, 2)
        x1 = 120
    vol = vol.transpose(tr)
    vol = vol[x1:-x1, ...]
    vol = np.rot90(vol, k=2, axes=(1, 2))
    vol = sp.util.resize(vol, [vol.shape[0], 128, vol.shape[2]])
    vol = resize(vol, [vol.shape[0], 512, 512])
    return vol

def process_case(fpath_data):
    data = np.load(fpath_data)
    data = data.transpose(0, 2, 1, 3, 4)
    data_sag = None
    data_cor = None
    for m in np.arange(data.shape[0]):
        for c in np.arange(data.shape[1]):
            vol = data[m, c]
            vol_sag = process_mpr(vol, plane='sag')
            if data_sag is None:
                data_sag = np.zeros((
                    data.shape[0], data.shape[1], vol_sag.shape[0], vol_sag.shape[1], vol_sag.shape[2]
                ))
            data_sag[m, c] = vol_sag

            vol_cor = process_mpr(vol, plane='cor')
            if data_cor is None:
                data_cor = np.zeros((
                    data.shape[0], data.shape[1], vol_cor.shape[0], vol_cor.shape[1], vol_cor.shape[2]
                ))

            data_cor[m, c] = vol_cor

    return data, data_sag, data_cor

def create_slices(vol, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for sl_idx in np.arange(vol.shape[2]):
        fname = os.path.join(dest_path, '{:03d}.npy'.format(sl_idx))
        np.save(fname, vol[:, :, sl_idx])

if __name__ == '__main__':
    dest_path = '/home/srivathsa/projects/studies/gad/mra_synth/preprocess/slices'
    src_path = '/home/srivathsa/projects/studies/gad/mra_synth/preprocess/data'
    cases = sorted([
        c.split('/')[-1].replace('.npy', '')
        for c in glob('{}/*.npy'.format(src_path))
    ])

    for cnum in tqdm(cases, total=len(cases)):
        fpath_npy = '{}/{}.npy'.format(src_path, cnum)
        data_ax, data_sag, data_cor = process_case(fpath_npy)
        create_slices(data_ax, '{}/{}/ax'.format(dest_path, cnum))
        create_slices(data_sag, '{}/{}/sag'.format(dest_path, cnum))
        create_slices(data_cor, '{}/{}/cor'.format(dest_path, cnum))
