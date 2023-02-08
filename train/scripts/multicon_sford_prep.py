from glob import glob
import subtle.utils.io as suio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (15, 12)

if __name__ == '__main__':
    fpath_t1 = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data'
    fpath_fl = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data_fl'
    fpath_uad = '/home/srivathsa/projects/studies/gad/stanford/preprocess/uad_fl'
    dest_path = '/home/srivathsa/projects/studies/gad/stanford/preprocess/data_512_tmp'

    cases = sorted([
        f.split('/')[-1].replace('.npy', '')
        for f in glob('{}/*.npy'.format(fpath_t1))
    ])

    cases = ['Patient_0089', 'Patient_0093', 'Patient_0112']

    ds_size = 512
    mean_norm = lambda v: v / v.mean()
    proc_fail = []
    for cnum in tqdm(cases, total=len(cases)):
        try:
            t1pre, t1low, t1post, t2 = suio.load_file(
                '{}/{}.npy'.format(fpath_t1, cnum), params={'h5_key': 'data'}
            ).transpose(1, 0, 2, 3)
            t1pre = mean_norm(t1pre)
            t1low = mean_norm(t1low)
            t1post = mean_norm(t1post)
            t2 = mean_norm(t2)

            _, _, _, fl = suio.load_file(
                '{}/{}.npy'.format(fpath_fl, cnum), params={'h5_key': 'data'}
            ).transpose(1, 0, 2, 3)

            fl = mean_norm(fl)

            # uad = np.load('{}/{}.npy'.format(fpath_uad, cnum))
            # th = uad.max() * 0.1
            # uad = (uad >= th)

            full_vol = np.array([t1pre, t1low, t1post, t2, fl, fl]).astype(np.float32)
            # vol_rs = resize(full_vol, (full_vol.shape[0], full_vol.shape[1] // 2, ds_size, ds_size))

            sl = full_vol.shape[1] // 2
            np.save('{}/{}.npy'.format(dest_path, cnum), full_vol[:5])

            row1 = np.hstack([full_vol[i][sl] for i in np.arange(3)])
            row2 = np.hstack([full_vol[i][sl] for i in np.arange(3, 6)])
            plt.imshow(np.vstack([row1, row2]))
            plt.axis('off')
            plt.savefig('{}/plots/{}.png'.format(dest_path, cnum))
            plt.clf()
        except Exception as ex:
            print('Cannot process {} - ERROR:{}'.format(cnum, ex))
            proc_fail.append(cnum)
