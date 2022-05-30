import argparse
import os
import glob
import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib
from torchvision.utils import save_image
from tqdm import tqdm
from configs.config import get_config
from networks.baseline_agis import MMT_baseline_agis as generator
from evaluator_brats import evaluator_brats, split_data
from evaluator_ixi import evaluator_ixi
from utils import list2str, make_image_grid


input_combination_brats = {'FLAIR': [0, 1, 2], 'T2': [0, 1, 3], 'T1Gd': [0, 2, 3], 'T1': [1, 2, 3]}

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mnt/raid/jiang/projects/SubtleGAN/data/brats2021_slices_crop192x160', help='root dir for data')
parser.add_argument('--cfg', type=str, default='configs/mmt.yml')
parser.add_argument('--dataset', type=str,
                    default='BRATS', help='experiment_name')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--contrast', type=str, default='T1Gd')
parser.add_argument('--save_dir', type=str, default='synthetic_images')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_contrast', type=int, default=4, help='total number of contrast in the dataset')
parser.add_argument('--zero_gad', action='store_true', default=False)


args = parser.parse_args()
            
            
def save_results(args, model, split='test', save_dir='vis'):
    data_dir = os.path.join(args.root_path, split)
    if args.dataset == 'BRATS':
        cases = glob.glob(f"{data_dir}/Bra*")
    elif args.dataset == 'IXI':
        cases = glob.glob(f"{data_dir}/IXI*")
    else:
        cases = glob.glob(f"{data_dir}/*")
    files = []
    # get all slices
    for case in tqdm(cases):
        files = glob.glob(f'{case}/*.npy')
        print(f"Generate {case.split('/')[-1]}")
        gt_contrasts = ['T1', 'T1Gd', 'T2', 'FLAIR']
        syn_contrasts = ['T1', 'T1Gd', 'T2', 'FLAIR'] if not args.zero_gad else ['T1Gd']
        gt_images = [[] for _ in range(len(gt_contrasts))]
        syn_images = [[] for _ in range(len(syn_contrasts))]
        for file in sorted(files):
            data = np.load(file)
            for i in range(args.n_contrast):
                gt_images[i].append(data[i])

            n_channel = data.shape[0]
            image = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).cuda()
            for i, contrast in enumerate(syn_contrasts):
                inputs = input_combination_brats[contrast]
                targets = list(set(range(args.n_contrast)) - set(inputs))

                data = [image[:, i, :, :].unsqueeze(0) for i in range(n_channel)]  # [(1, 1, H, W)]
                img_inputs, img_targets, contrast_input, contrast_output = split_data(data, inputs, targets)
                img_outputs = model(img_inputs, contrast_input, contrast_output)
                img_output = img_outputs[0].detach().cpu().numpy()[0][0]
                syn_images[i].append(img_output)

        # save results
        case_name = case.split("/")[-1]
        save_dir = os.path.join(args.model_path, args.save_dir, split, case_name)
        os.makedirs(save_dir, exist_ok=True)
        for i, gt_image in enumerate(gt_images):
            gt_image = nib.Nifti1Image(np.stack(gt_image, axis=-1), np.eye(4))
            fn = os.path.join(save_dir, f'{gt_contrasts[i]}.nii.gz')
            nib.save(gt_image, fn)
            print(fn)

        for i, syn_image in enumerate(syn_images):
            syn_image = nib.Nifti1Image(np.stack(syn_image, axis=-1), np.eye(4))
            fn = os.path.join(save_dir, f'{syn_contrasts[i]}_syn.nii.gz')
            nib.save(syn_image, fn)
            print(fn)



if __name__ == "__main__":
    G = generator(n_contrast=args.n_contrast).cuda()

    state_dict = torch.load(os.path.join(args.model_path, args.ckpt), map_location='cpu')
    G.load_state_dict(state_dict['G'])
    G.eval()

    save_results(args, G, split='test', save_dir=args.save_dir)


