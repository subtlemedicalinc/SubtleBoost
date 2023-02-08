import argparse
import os
from glob import glob
import numpy as np
import torch
from networks.mmt import MMT as generator
from torchvision.utils import save_image
from tqdm import tqdm
from configs.config import get_config
from evaluator import evaluator_brats, evaluator_ixi, evaluator_spine, split_data, generate_spine_images
from utils import list2str, make_image_grid
from scipy.ndimage import zoom
import SimpleITK as sitk
from subtle.subtle_preprocess import dcm_to_sitk
from subtle.utils.io import write_dicoms

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, help='root dir for data')
parser.add_argument('--cfg', type=str, default='configs/mmt.yml')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--split', type=str, help='Dataset split', default='test')
parser.add_argument('--dicom_out', type=str, help='DICOM output folder')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument(
    '--raw_data_path', type=str,
    default='/home/srivathsa/projects/studies/gad/stanford/data',
    help='Path to raw DICOM files'
)
parser.add_argument('--mra_synth', action='store_true', default=False)

args = parser.parse_args()

def get_output_vol(model, inputs, targets, files, model_path, split, vis_dir='vis'):
    vol = []
    model.eval()
    with torch.no_grad():
        for file in files:
            data = np.load(file)
            n_channel = data.shape[0]
            image = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).cuda()
            data = [image[:, i, :, :].unsqueeze(0) for i in range(n_channel)]  # [(1, 1, H, W)]
            img_inputs, img_targets, contrast_input, contrast_output = split_data(data, inputs, targets)
            img_outputs, _, _ = model(img_inputs, contrast_input, contrast_output)
            out_slice = img_outputs[0][0, 0].cpu().detach().numpy()
            vol.append(out_slice)

    vol = np.array(vol)
    return vol

def postprocess(data, dirpath_ref, rs_dim=256, x1=24, x2=8, y1=32, y2=32):
    ref_data = sitk.GetArrayFromImage(dcm_to_sitk(dirpath_ref))

    if data.shape[1] != rs_dim:
        # rotate back
        data = np.rot90(data, axes=(1, 2), k=3)

        # undo cropping
        data = np.pad(data, pad_width=[(0, 0), (x1, x2), (y1, y2)])

        # undo resampling from 256 -> 224
        zf = rs_dim / data.shape[1]
        data = zoom(data, (1, zf, zf))

    zf_dcm = np.divide(ref_data.shape, data.shape)
    data = zoom(data, zf_dcm)

    data = np.interp(data, (data.min(), data.max()), (ref_data.min(), ref_data.max()))
    return data

def postprocess_ixi(data, dirpath_ref):
    data = np.rot90(data, axes=(1, 2))
    data = zoom(data, (1, 2, 2))
    ref_data = sitk.GetArrayFromImage(dcm_to_sitk(dirpath_ref))
    data = np.interp(data, (data.min(), data.max()), (ref_data.min(), ref_data.max()))
    return data

def process_single(args, model, case_number, inputs, output, series_desc, dicom_template):
    npy_files = sorted([
        f for f in glob('{}/{}/{}/*.npy'.format(args.root_path, args.split, case_number))
    ])

    output_vol = get_output_vol(
        model=model, inputs=inputs, targets=output, files=npy_files,
        model_path=args.model_path, split=args.split
    )

    if args.mra_synth:
        output_vol = postprocess_ixi(output_vol, dicom_template)
    else:
        output_vol = postprocess(output_vol, dicom_template)

    dirpath_dicom_out = os.path.join(args.dicom_out, case_number, series_desc)
    write_dicoms(
        dicom_template, output_vol, dirpath_dicom_out,
        series_desc_pre='SubtleMDI Scratch: ', desc=series_desc, series_desc_post=''
    )

def find_dicom_template(dirpath):
    dirlist = [d for d in glob('{}/*'.format(dirpath))]
    dirlist = sorted(dirlist, key=lambda d: int(d.split('/')[-1].split('_')[0]))
    return dirlist[0]

if __name__ == "__main__":

    if args.mra_synth:
        series_desc_map = {
            3: 'mra_synth'
        }
        comb_list = [[0, 1]]
    else:
        series_desc_map = {
            0: 't1_synth',
            1: 't1_gad_synth',
            2: 't2_synth',
            3: 'flair_synth'
        }
        comb_list = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    config = get_config(args)
    G = generator(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                            in_chans=config.MODEL.SWIN.IN_CHANS,
                            out_chans = config.MODEL.SWIN.OUT_CHANS,
                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                            depths=config.MODEL.SWIN.DEPTHS,
                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=config.MODEL.SWIN.APE,
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                            seg=False,
                            num_contrast=4).cuda()

    state_dict = torch.load(os.path.join(args.model_path, args.ckpt), map_location='cpu')
    G.load_state_dict(state_dict['G'])
    G.eval()

    cases = sorted([c.split('/')[-1] for c in glob('{}/{}/*'.format(args.root_path, args.split))])
    for case_number in tqdm(cases, total=len(cases)):
        dicom_template = find_dicom_template(os.path.join(args.raw_data_path, case_number))

        for inputs in comb_list:
            if args.mra_synth:
                output = [3]
            else:
                output = [c for c in np.arange(4) if c not in inputs]
            series_desc = series_desc_map[output[0]]
            process_single(args, G, case_number, inputs, output, series_desc, dicom_template)
