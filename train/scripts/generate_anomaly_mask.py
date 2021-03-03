import os
import tempfile
from glob import glob
from datetime import datetime

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from skimage.morphology import ball
from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
import tensorflow as tf

from subtle.subtle_preprocess import dcm_to_sitk, center_crop
import subtle.utils.io as suio
from ventmapper.segment.ventmapper import process_vent

from models import variational_autoencoder
from utils.default_config_setup import get_config, get_options, Dataset, get_Brainweb_healthy_dataset
from trainers.VAE import VAE
from utils.Evaluation import apply_brainmask, normalize_and_squeeze
from skimage import color
import pydicom

plt.set_cmap('gray')
plt.rcParams['figure.figsize'] = (12, 10)

pp_base_path = '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data_fl'
raw_base_path = '/home/srivathsa/projects/studies/gad/tiantan/data'
plot_path_template = '/home/srivathsa/projects/uad_ae/plots/fl/{case_num}_{sl}.png'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
t2_quantile = 0.99

def find_pre_contrast_series(dirpath_case):
    ser_nums = []
    for ser_dir in glob('{}/*'.format(dirpath_case)):
        if 'T2' in ser_dir or 'FLAIR' in ser_dir: continue
        dcm_file = [
            f for f in glob('{}/**/*'.format(ser_dir), recursive=True)
            if os.path.isfile(f) and 'XX' not in f
        ][0]
        dcm = pydicom.dcmread(dcm_file)
        ser_nums.append((ser_dir, dcm.SeriesNumber))

    ser_nums = sorted(ser_nums, key=lambda s: int(s[1]))
    return ser_nums[0][0]

def resize_image(input_sitk, new_spacing=[1, 1, 1]):
    orig_spacing = input_sitk.GetSpacing()
    orig_size = input_sitk.GetSize()

    res_filter = sitk.ResampleImageFilter()
    res_filter.SetInterpolator(sitk.sitkLinear)
    res_filter.SetOutputDirection(input_sitk.GetDirection())
    res_filter.SetOutputOrigin(input_sitk.GetOrigin())
    res_filter.SetOutputSpacing(new_spacing)
    rs_size = np.array(orig_spacing) * np.array(input_sitk.GetSize())
    res_filter.SetSize([int(rs_size[0]), int(rs_size[1]), int(rs_size[2])])
    return res_filter.Execute(input_sitk), orig_size

def process_and_return_sitk(case_num, full_data):
    t1_sitk = dcm_to_sitk(find_pre_contrast_series('{}/{}'.format(raw_base_path, case_num)))
    t1_mask = full_data[1, :, 0]
#     t1_mask = np.load('{}/{}.npy'.format(pp_base_path, case_num))[1, :, 0]
    t1_mask = binary_fill_holes(t1_mask > 0.1)
    t1_mask_sitk = sitk.GetImageFromArray(t1_mask.astype(np.uint8))
    t1_mask_sitk.CopyInformation(t1_sitk)

    t1_rs, orig_size = resize_image(t1_sitk)
    t1m_rs, _ = resize_image(t1_mask_sitk)

    return t1_rs, t1m_rs, orig_size

def get_vent_mask(t1_stk, t1m_stk):
    img = None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = '/home/srivathsa/projects'
        fpath_input = '{}/input.nii.gz'.format(tmpdir)
        fpath_input_mask = '{}/input-mask.nii.gz'.format(tmpdir)
        sitk.WriteImage(t1_stk, fpath_input)
        sitk.WriteImage(t1m_stk, fpath_input_mask)

        img = process_vent(
            subj_dir='', subj='', t1=fpath_input, fl=None, t2=None,
            mask=fpath_input_mask, out='dummy', force=True, return_only=True
        )
    return img.get_data()

def load_uad_model():
    dataset = Dataset.BRAINWEB
    options = get_options(batchsize=8, learningrate=0.0001, numEpochs=100, zDim=128, outputWidth=128, outputHeight=128, slices_start=0, slices_end=200)
    options['data']['dir'] = options["globals"][dataset.value]
    dataset = get_Brainweb_healthy_dataset(options)
    config = get_config(trainer=VAE, options=options, optimizer='ADAM', intermediateResolutions=[8, 8], dropout_rate=0.1, dataset=dataset)

    # Create an instance of the model and train it
    model = VAE(tf.Session(), config, network=variational_autoencoder.variational_autoencoder)
    model.load_checkpoint()
    return model

def segment_anomalies(vae_model, case_num, full_data, prune_final_mask=False, prune_quant=0.95):
    t2_data = full_data[1, :, 3]
#     t2_data = np.load('{}/{}.npy'.format(pp_base_path, case_num))[1, :, 3]
    if t2_data.shape[1] != 256:
        t2_data = t2_data.transpose(1, 2, 0)
        pw1 = (256 - t2_data.shape[1]) // 2
        pw2 = (256 - t2_data.shape[2]) // 2
        t2_data = np.pad(t2_data, pad_width=[(0, 0), (pw1, pw1), (pw2, pw2)])

    t2_data = resize(t2_data, (t2_data.shape[0], 128, 128))
    t2_data = np.clip(t2_data, 0, t2_data.max())
    t2_data = np.interp(t2_data, (t2_data.min(), t2_data.max()), (0, 1))[..., None]

    t2_recon = vae_model.reconstruct(t2_data)['reconstruction']
    x_diff = np.maximum(t2_data - t2_recon, 0)
#     x_diff = np.abs(t2_data - t2_recon)
    brainmask = binary_fill_holes(t2_data > 0.1)
    prior_quantile = np.quantile(t2_data, t2_quantile)

    for sl_idx in range(x_diff.shape[0]):
        diff_sl = x_diff[sl_idx, ..., 0]
        diff_sl = apply_brainmask(diff_sl, brainmask[sl_idx, ..., 0])
        diff_sl[t2_data[sl_idx, ..., 0] < prior_quantile] = 0
        diff_sl = median_filter(normalize_and_squeeze(diff_sl), (5, 5))
        x_diff[sl_idx, ..., 0] = diff_sl

    uad_mask = resize(np.squeeze(x_diff), (x_diff.shape[0], 256, 256))

    if full_data.shape[-1] != 256:
        uad_mask = uad_mask.transpose(2, 0, 1)
        uad_mask = center_crop(uad_mask, np.zeros((full_data.shape[1], full_data.shape[3], full_data.shape[4])))

    if prune_final_mask:
        mask_quant = np.quantile(uad_mask, prune_quant)
        uad_mask[uad_mask < mask_quant] = 0

    return uad_mask

def format_vent_mask(mask, orig_size):
    mask_rs = mask.transpose(2, 1, 0)

    orig_shape = orig_size[::-1]
    mask_rs = np.pad(mask_rs, pad_width=[
        (0, orig_shape[0]-mask_rs.shape[0]), (0, orig_shape[1]-mask_rs.shape[1]), (0, orig_shape[2]-mask_rs.shape[2])
    ])

    mask_rs = binary_fill_holes(mask_rs)
    mask_rs = binary_dilation(mask_rs, iterations=3).astype(np.uint8)
    mask_rs = np.interp(mask_rs, (mask_rs.min(), mask_rs.max()), (0, 1))

    return mask_rs

def get_rgb(img):
    img = (img - np.min(img))/np.ptp(img)
    return np.dstack((img, img, img))

def get_anomalies(case_num, vae_model, prune_final_mask=False, remove_vent=True):
    full_data = suio.load_file('{}/{}.h5'.format(pp_base_path, case_num), params={'h5_key': 'all'})

    t1_stk, t1m_stk, orig_size = process_and_return_sitk(case_num, full_data)


    uad_mask = segment_anomalies(vae_model, case_num, full_data, prune_final_mask)
    final_mask = uad_mask.copy()

    if remove_vent:
        vent_mask = get_vent_mask(t1_stk, t1m_stk)
        vent_mask = format_vent_mask(vent_mask, orig_size)
        final_mask[vent_mask.astype(np.uint8) == 1] = 0

    final_mask = gaussian_filter(final_mask, sigma=2)
    final_mask = resize(final_mask, (final_mask.shape[0], full_data.shape[3], full_data.shape[4]))
    final_mask = np.interp(final_mask, (final_mask.min(), final_mask.max()), (0, 1))

    return final_mask, full_data

def overlay_mask(data, label, r=0.2, g=1.0, b=0.2):
    data_rgb = get_rgb(data)

    label_r = label * r
    label_g = label * g
    label_b = label * b
    label_rgb = np.dstack((label_r, label_g, label_b))

    data_hsv = color.rgb2hsv(data_rgb)
    label_hsv = color.rgb2hsv(label_rgb)

    data_hsv[..., 0] = label_hsv[..., 0]
    data_hsv[..., 1] = label_hsv[..., 1] * 0.55

    return color.hsv2rgb(data_hsv)


def segviz(image, label, sl=None, title=None, fpath_plot=None, tparams=(1.0, 0.0)):
    if not sl:
        sl = image.shape[0] // 2

    data_sl = image[sl]
    label_sl = label[sl]

    img_overlay = overlay_mask(data_sl, label_sl)

    plt.imshow((tparams[0] * img_overlay) + tparams[1], vmin=img_overlay.min(), vmax=img_overlay.max())
    plt.axis('off')
    if title:
        plt.title(title, size=16)
    if fpath_plot:
        plt.savefig(fpath_plot)
        plt.clf()
    else:
        plt.show()

# case_info = [
#     ('Patient_0187', 171),
#     ('Patient_0351', 191),
#     ('Patient_0378', 72),
#     ('Patient_0470', 75),
#     ('Patient_0492', 48),
#     ('Patient_0492', 92),
#     ('Patient_0495', 85)
# ]

if __name__ == '__main__':
    ignore_cases = [
        c.split('/')[-1].replace('.npy', '')
        for c in glob('{}/*.npy'.format(pp_base_path.replace('data', 'uad_masks')))
    ]

    # cases = sorted([c.split('/')[-1] for c in glob('/mnt/datasets/srivathsa/tiantan/Batch1/*')])
    # cases = sorted([c.split('/')[-1].replace('.npy', '') for c in glob('/home/srivathsa/projects/studies/gad/tiantan/preprocess/uad_masks/no_prune/*.npy')])

    cases = sorted([
        c.split('/')[-1].replace('.h5', '')
        for c in glob('{}/*.h5'.format(pp_base_path))
        if 'meta' not in c and 'Prisma' not in c and 'TwoDim' not in c
    ])

    def sort_key(s):
        if 'NO' in s:
            return int(s.replace('NO', ''))
        elif 'Brain' in s:
            return 0
        else:
            return 1

    cases = sorted([c for c in cases if c not in ignore_cases], key=sort_key)

    # cases = ['NO31']

    tf.reset_default_graph()
    vae_model = load_uad_model()
    for case_num in cases:
        try:
            start = datetime.now()

            uad_mask, full_data = get_anomalies(case_num, vae_model, prune_final_mask=False, remove_vent=False)
            end = datetime.now()
            print('time elapsed for {} = {}'.format(case_num, end-start))

            fpath_mask = os.path.join(
                pp_base_path.replace('data', 'uad_masks'), '{}.npy'.format(case_num)
            )
            np.save(fpath_mask, uad_mask)
        except Exception as ex:
            print('Cannot process {}: Error:{}'.format(case_num, ex))

        # th = uad_mask.max() * 0.1
        # uad_mask_disp = uad_mask >= th
        # segviz(full_data[1, :, 2], uad_mask_disp, sl=sl, fpath_plot=plot_path_template.format(case_num=case_num, sl=sl))
