import os
import numpy as np
from glob import glob
from skimage.transform import resize
import sigpy as sp
from tqdm import tqdm
import subtle.utils.io as suio
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_erosion
from subtle.subtle_preprocess import enh_mask_smooth
from multiprocessing import Pool

import matplotlib.pyplot as plt
plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (10, 8)

def process_mpr_ax(vol, plane, rs=240):
    if plane == 'sag':
        tr = (2, 0, 1)
        x1 = 90
    else:
        tr = (1, 0, 2)
        x1 = 60
    vol = vol.transpose(tr)
    vol = np.rot90(vol, k=2, axes=(1, 2))
    if vol.shape[1] / rs < 0.5:
        rs_tmp = rs // 2
        vol = sp.util.resize(vol, [vol.shape[0], rs_tmp, vol.shape[2]])
        vol = resize(vol, [vol.shape[0], rs, vol.shape[2]])
    else:
        vol = sp.util.resize(vol, [vol.shape[0], rs, vol.shape[2]])

    vol = vol[x1:-x1]
    return vol

def remove_noisy_bg(vol, th=1e-3):
    all_vols = []
    for m in np.arange(vol.shape[0]):
        pre, low, post = vol[m]

        pre_mask = (pre > th)
        pre_mask = binary_fill_holes(pre_mask)

        pre = pre * pre_mask
        low = low * pre_mask
        post = post * pre_mask

        all_vols.append(np.array([pre, low, post]))

    return np.array(all_vols)

def process_mpr_sag(vol, plane, rs=240):
    if plane == 'ax':
        tr = (1, 0, 2)
        x1 = 0
    else:
        tr = (2, 0, 1)
        x1 = 0
    vol = vol.transpose(tr)
    vol = np.rot90(vol, k=3, axes=(1, 2))
    if vol.shape[2] / rs < 0.5:
        rs_tmp = rs // 2
        vol = sp.util.resize(vol, [vol.shape[0], vol.shape[1], rs_tmp])
        vol = resize(vol, [vol.shape[0], vol.shape[1], rs])
    else:
        vol = sp.util.resize(vol, [vol.shape[0], rs, rs])
    return vol

def process_case(fpath_data, acq_plane, rs_dim):
    data = suio.load_file(fpath_data, params={'h5_key': 'all'})
    data = data.transpose(0, 2, 1, 3, 4)
    data = np.clip(data, 0, data.max())

    data_or1 = None
    data_or2 = None

    all_planes = ['ax', 'sag', 'cor']
    planes = [p for p in all_planes if p != acq_plane]

    if acq_plane == 'ax':
        proc_fn = process_mpr_ax
    elif acq_plane == 'sag':
        proc_fn = process_mpr_sag

    data = data[:, [0, 1, 2]]

    for m in np.arange(data.shape[0]):
        for c in np.arange(data.shape[1]):
            vol = data[m, c]
            vol_or1 = proc_fn(vol, plane=planes[0], rs=rs_dim)
            if data_or1 is None:
                data_or1 = np.zeros((
                    data.shape[0], data.shape[1], vol_or1.shape[0], vol_or1.shape[1], vol_or1.shape[2]
                ))
            data_or1[m, c] = vol_or1

            vol_or2 = proc_fn(vol, plane=planes[1], rs=rs_dim)
            if data_or2 is None:
                data_or2 = np.zeros((
                    data.shape[0], data.shape[1], vol_or2.shape[0], vol_or2.shape[1], vol_or2.shape[2]
                ))

            data_or2[m, c] = vol_or2

    if acq_plane == 'ax':
        return data, data_or1, data_or2

    return data_or1, data, data_or2


def create_slices(vol, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    enh_mask_vol = []

    for sl_idx in np.arange(vol.shape[2]):
        fname = os.path.join(dest_path, '{:03d}.npy'.format(sl_idx))
        vol_slice = vol[:, :, sl_idx]
        X_inp = vol_slice[1, [0, 1]][None, None]
        Y_inp = vol_slice[1, [2]][None, None]

        enh_mask = enh_mask_smooth(X_inp, Y_inp, center_slice=0).squeeze()
        enh_mask = np.nan_to_num(enh_mask, nan=0.01)
        enh_mask = np.clip(enh_mask, 0.01, vol[1, 2].max())
        enh_mask_vol.append(enh_mask)

        save_npy = np.array([*vol_slice[0]] + [enh_mask])
        np.save(fname, save_npy)

    return enh_mask_vol

def plot_slices(vol, enh_mask, plot_path, cnum, suffix):
    fpath_png = '{}/{}{}.png'.format(plot_path, cnum, suffix)

    pre, low, full = vol[0]
    sl = pre.shape[0] // 2

    row1 = np.hstack([pre[sl], low[sl], full[sl]])
    row2 = np.hstack([low[sl]-pre[sl], full[sl]-pre[sl], enh_mask[sl]])
    img = np.vstack([row1, row2])
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.imshow(row1)
    ax1.axis('off')

    ax2.imshow(row2)
    ax2.axis('off')

    plt.title(cnum)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.margins(y=0)
    plt.savefig(fpath_png, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()

def process_single_case(params):
    try:
        src_path = params['src_path']
        dest_path = params['dest_path']
        file_ext = params['file_ext']
        acq_plane = params['acq_plane']
        cnum = params['cnum']
        rs_dim = params['rs_dim']
        plot_path = params['plot_path']

        print('Processing {}...'.format(cnum))
        fpath_data = '{}/{}.{}'.format(src_path, cnum, file_ext)
        data_ax, data_sag, data_cor = process_case(fpath_data, acq_plane=acq_plane, rs_dim=rs_dim)

        enh_mask_ax = create_slices(data_ax, '{}/{}/ax'.format(dest_path, cnum))
        enh_mask_sag = create_slices(data_sag, '{}/{}/sag'.format(dest_path, cnum))
        enh_mask_cor = create_slices(data_cor, '{}/{}/cor'.format(dest_path, cnum))

        plot_slices(data_ax, enh_mask_ax, plot_path, cnum, suffix='')
        plot_slices(data_sag, enh_mask_sag, plot_path, cnum, suffix='_sag')
        plot_slices(data_cor, enh_mask_cor, plot_path, cnum, suffix='_cor')
    except Exception as exc:
        print('ERROR in {} - {}'.format(cnum, exc))

def get_acq_plane(case_num):
    kw_plane_map = {
        'Id': 'sag',
        'Patient': 'ax',
        'NO': 'sag',
        'Brain': 'sag',
        'Prisma': 'sag'
    }

    for kw, acq_plane in kw_plane_map.items():
        if kw in case_num:
            return acq_plane

    raise ValueError('Cannot find acquisition plane for Case:{}'.format(case_num))

if __name__ == '__main__':
    dest_path = '/mnt/local_datasets/srivathsa/all/preprocess/slices'
    src_path = '/home/srivathsa/projects/studies/gad/all/preprocess/data'
    plot_path = '/mnt/local_datasets/srivathsa/all/preprocess/slices/plots'
    file_ext = 'npy'
    rs_dim = 512

    cases = sorted([
        c.split('/')[-1].replace(f'.{file_ext}', '')
        for c in glob('{}/*.{}'.format(src_path, file_ext))
        if 'meta' not in c and 'TwoDim' not in c
    ])

    cases = [
        "Brain2H-600441599", "Brain4H-601044594"
    ]

    proc_params = []
    for cnum in cases:
        proc_params.append({
            'src_path': src_path,
            'dest_path': dest_path,
            'plot_path': plot_path,
            'file_ext': file_ext,
            'cnum':cnum,
            'acq_plane': get_acq_plane(cnum),
            'rs_dim': rs_dim
        })

    process_pool = Pool(processes=16, initializer=None, initargs=None)
    _ = process_pool.map(process_single_case, proc_params)
    process_pool.close()
    process_pool.join()
