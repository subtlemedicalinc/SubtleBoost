# conda activate simpleitk
import SimpleITK as sitk
import nibabel as nib
import time
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from subtle.subtle_preprocess import register_im
import matplotlib.pyplot as plt
plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (15, 12)

def plot_ixi(dir, case, plot_path):
    cons = ['T1', 'T2', 'PD', 'MRA']
    imgs = []

    for c in cons:
        vol = nib.load('{}/{}/{}-{}.nii.gz'.format(dir, case, case, c)).get_fdata()
        sl = vol.shape[-1] // 2
        imgs.append(vol[..., sl])

    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols)

    k = 0
    for i in np.arange(nrows):
        for j in np.arange(ncols):
            ax[i][j].imshow(imgs[k])
            ax[i][j].axis('off')
            k += 1

    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    fpath_plot = '{}/{}.png'.format(plot_path, case)
    plt.savefig(fpath_plot)
    plt.clf()

src_dir = '/mnt/datasets/srivathsa/jiang_raid/projects/SubtleGAN/data/IXI/IXI_dataset'
data_dir = '/mnt/datasets/srivathsa/jiang_raid/projects/SubtleGAN/data/IXI/IXI_MRA/coreg'
plot_dir = '/mnt/datasets/srivathsa/jiang_raid/projects/SubtleGAN/data/IXI/IXI_MRA/coreg_plots'

cases = sorted([c.split('/')[-1] for c in glob('{}/IXI*'.format(src_dir))])
pmap = sitk.GetDefaultParameterMap('affine')

for case in tqdm(cases, total=len(cases)):
    case_name = case.split("/")[-1]
    case_dir = os.path.join(data_dir, case)

    t1_file = '{}/{}/{}-T1.nii.gz'.format(src_dir, case, case)
    t2_file = '{}/{}/{}-T2.nii.gz'.format(src_dir, case, case)
    pd_file = '{}/{}/{}-PD.nii.gz'.format(src_dir, case, case)
    mra_file = '{}/{}/{}-MRA.nii.gz'.format(src_dir, case, case)

    fcount = np.sum([
        1 if os.path.exists(f) else 0
        for f in [t1_file, t2_file, pd_file, mra_file
    ]])
    if fcount != 4:
        continue

    t1_stk = sitk.ReadImage(t1_file)
    t2_stk = sitk.ReadImage(t2_file)
    pd_stk = sitk.ReadImage(pd_file)
    mra_stk = sitk.ReadImage(mra_file)

    t1_reg, _ = register_im(
        im_fixed=sitk.GetArrayFromImage(mra_stk), im_moving=sitk.GetArrayFromImage(t1_stk),
        ref_fixed=mra_stk, ref_moving=t1_stk, param_map=pmap, return_sitk_img=True, verbose=False
    )
    t2_reg, _ = register_im(
        im_fixed=sitk.GetArrayFromImage(mra_stk), im_moving=sitk.GetArrayFromImage(t2_stk),
        ref_fixed=mra_stk, ref_moving=t2_stk, param_map=pmap, return_sitk_img=True, verbose=False
    )
    pd_reg, _ = register_im(
        im_fixed=sitk.GetArrayFromImage(mra_stk), im_moving=sitk.GetArrayFromImage(pd_stk),
        ref_fixed=mra_stk, ref_moving=pd_stk, param_map=pmap, return_sitk_img=True, verbose=False
    )

    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    t1_dest = os.path.join(case_dir, '{}-T1.nii.gz'.format(case))
    t2_dest = os.path.join(case_dir, '{}-T2.nii.gz'.format(case))
    pd_dest = os.path.join(case_dir, '{}-PD.nii.gz'.format(case))
    mra_dest = os.path.join(case_dir, '{}-MRA.nii.gz'.format(case))

    sitk.WriteImage(t1_reg, t1_dest)
    sitk.WriteImage(t2_reg, t2_dest)
    sitk.WriteImage(pd_reg, pd_dest)
    sitk.WriteImage(mra_stk, mra_dest)

    plot_ixi(data_dir, case, plot_dir)
