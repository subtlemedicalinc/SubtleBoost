# conda activate simpleitk
import SimpleITK as sitk
import nibabel as nib
import time
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from medpy.io import load as load_mha

from subtle.subtle_preprocess import register_im
import matplotlib.pyplot as plt

plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (15, 12)

def mha2sitk(fpath_mha):
    img, hdr = load_mha(fpath_mha)
    return hdr.get_sitkimage()

def plot_tubetk(dir, case, plot_path):
    cons = ['T1', 'T2', 'MRA']
    imgs = []

    for c in cons:
        vol = nib.load('{}/{}/{}_{}.nii.gz'.format(dir, case, case, c)).get_fdata()
        sl = vol.shape[-1] // 2
        imgs.append(vol[..., sl])

    nrows = 1
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols)

    k = 0
    for i in np.arange(nrows):
        for j in np.arange(ncols):
            ax[j].imshow(imgs[k])
            ax[j].axis('off')
            k += 1

    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    fpath_plot = '{}/{}.png'.format(plot_path, case)
    plt.savefig(fpath_plot)
    plt.clf()

src_dir = '/mnt/datasets/srivathsa/tubetk_mra/raw'
dest_dir = '/mnt/datasets/srivathsa/tubetk_mra/reg'
plot_dir = '/mnt/datasets/srivathsa/tubetk_mra/plots_reg'

cases = sorted([c.split('/')[-1] for c in glob('{}/TubeTK*'.format(src_dir))])

proc_cases = sorted([
    c.split('/')[-1].replace('.png', '')
    for c in glob('{}/*.png'.format(plot_dir))
])

cases = [c for c in cases if c not in proc_cases]
pmap = sitk.GetDefaultParameterMap('affine')

for case in tqdm(cases, total=len(cases)):
    case_dir = os.path.join(src_dir, case)

    fpath_t1 = os.path.join(case_dir, '{}_T1MPRage.mha'.format(case))
    fpath_t2 = os.path.join(case_dir, '{}_T2.mha'.format(case))
    fpath_mra = os.path.join(case_dir, '{}_MRA.mha'.format(case))
    if not os.path.exists(fpath_t1) or not os.path.exists(fpath_t2) or not os.path.exists(fpath_mra):
        print('Skipping {}...'.format(case))
        continue

    t1_stk = mha2sitk(fpath_t1)
    t2_stk = mha2sitk(fpath_t2)
    mra_stk = mha2sitk(fpath_mra)

    t2_reg, _ = register_im(
        im_fixed=sitk.GetArrayFromImage(t1_stk), im_moving=sitk.GetArrayFromImage(t2_stk),
        ref_fixed=t1_stk, ref_moving=t2_stk, param_map=pmap, return_sitk_img=True, verbose=False
    )

    mra_reg, _ = register_im(
        im_fixed=sitk.GetArrayFromImage(t1_stk), im_moving=sitk.GetArrayFromImage(mra_stk),
        ref_fixed=t1_stk, ref_moving=mra_stk, param_map=pmap, return_sitk_img=True, verbose=False
    )

    out_dir = os.path.join(dest_dir, case)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sitk.WriteImage(t1_stk, '{}/{}_T1.nii.gz'.format(out_dir, case))
    sitk.WriteImage(t2_reg, '{}/{}_T2.nii.gz'.format(out_dir, case))
    sitk.WriteImage(mra_reg, '{}/{}_MRA.nii.gz'.format(out_dir, case))

    plot_tubetk(dest_dir, case, plot_dir)
