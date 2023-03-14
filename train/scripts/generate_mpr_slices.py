import os
import numpy as np
from glob import glob
from skimage.transform import resize
import sigpy as sp
from tqdm import tqdm
import subtle.utils.io as suio
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_erosion

def process_mpr_mra(vol, plane):
    if plane == 'sag':
        tr = (2, 0, 1)
        x1 = 0
    else:
        tr = (1, 0, 2)
        x1 = 0
    vol = vol.transpose(tr)
    vol = vol[:, ...]
    vol = np.rot90(vol, k=2, axes=(1, 2))
    vol = sp.util.resize(vol, [vol.shape[0], 128, vol.shape[2]])
    vol = resize(vol, [vol.shape[0], 512, 512])
    return vol

def remove_noisy_bg(img_vol, th=0.5):
    img_mask = img_vol >= th
    img_mask = binary_erosion(img_mask)
    img_mask = binary_fill_holes(img_mask)
    return img_vol * img_mask

def process_mpr_tiantan(vol, plane):
    if plane == 'ax':
        tr = (1, 0, 2)
        x1 = 0
    else:
        tr = (2, 0, 1)
        x1 = 0
    vol = vol.transpose(tr)
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = sp.util.resize(vol, [vol.shape[0], 240, 240])
    # vol = remove_noisy_bg(vol)
    return vol

def process_case(fpath_data, mode='mra'):
    data = suio.load_file(fpath_data, params={'h5_key': 'all'})
    data = data.transpose(0, 2, 1, 3, 4)

    if 'Prisma' in fpath_data:
        data = data[..., 8:-8, 8:-8]

    data_or1 = None
    data_or2 = None

    if mode == 'mra':
        proc_fn = process_mpr_mra
        planes = ['sag', 'cor']
    elif mode == 'tiantan':
        proc_fn = process_mpr_tiantan
        planes = ['ax', 'cor']
        data = data[:, [0, 1, 2]]

    for m in np.arange(data.shape[0]):
        for c in np.arange(data.shape[1]):
            vol = data[m, c]
            vol_or1 = proc_fn(vol, plane=planes[0])
            if data_or1 is None:
                data_or1 = np.zeros((
                    data.shape[0], data.shape[1], vol_or1.shape[0], vol_or1.shape[1], vol_or1.shape[2]
                ))
            data_or1[m, c] = vol_or1

            vol_or2 = proc_fn(vol, plane=planes[1])
            if data_or2 is None:
                data_or2 = np.zeros((
                    data.shape[0], data.shape[1], vol_or2.shape[0], vol_or2.shape[1], vol_or2.shape[2]
                ))

            data_or2[m, c] = vol_or2

    if mode == 'mra':
        return data, data_or1, data_or2
    elif mode == 'tiantan':
        return data_or1, data, data_or2


def create_slices(vol, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for sl_idx in np.arange(vol.shape[2]):
        fname = os.path.join(dest_path, '{:03d}.npy'.format(sl_idx))
        np.save(fname, vol[:, :, sl_idx])

if __name__ == '__main__':
    dest_path = '/home/srivathsa/projects/studies/gad/tiantan/preprocess/slices'
    src_path = '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data'

    mode = 'tiantan'
    file_ext = 'h5'

    cases = sorted([
        c.split('/')[-1].replace(f'.{file_ext}', '')
        for c in glob('{}/*.{}'.format(src_path, file_ext))
        if 'meta' not in c and 'TwoDim' not in c
    ])

    for cnum in tqdm(cases, total=len(cases)):
        try:
            fpath_data = '{}/{}.{}'.format(src_path, cnum, file_ext)
            data_ax, data_sag, data_cor = process_case(fpath_data, mode=mode)
            create_slices(data_ax, '{}/{}/ax'.format(dest_path, cnum))
            create_slices(data_sag, '{}/{}/sag'.format(dest_path, cnum))
            create_slices(data_cor, '{}/{}/cor'.format(dest_path, cnum))
        except Exception as exc:
            print('ERROR in {} - {}'.format(cnum, exc))
