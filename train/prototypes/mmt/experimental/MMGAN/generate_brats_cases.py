from modules.advanced_gans.models import *
from modules.helpers import create_dataloaders, Resize, ToTensor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import nibabel as nib
import itertools
import torch
import os
import torch
import argparse
import pdb
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_contrast', type=int, default=3)
parser.add_argument('--n_slices', type=int, default=90)
parser.add_argument('--n_cases', type=int, default=30)
parser.add_argument('--ckpt', type=str, default="model/MMGAN/mmgan_ixi_single/generator_param_mmgan_ixi_single_60.pkl")
parser.add_argument('--data_path', type=str, default="/mnt/raid/jiang/projects/SubtleGAN/data/IXI/hdf5/")
parser.add_argument('--model_path', type=str, default="model/MMGAN/mmgan_ixi_single/")
parser.add_argument('--save_dir', type=str, default="synthetic_images")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
generator = GeneratorUNet(in_channels=args.n_contrast, out_channels=args.n_contrast, with_relu=True, with_tanh=False)
ckpt = torch.load(args.ckpt, map_location='cpu')
generator = nn.DataParallel(generator.cuda())
generator.load_state_dict(ckpt['state_dict'])

n_dataloader, dataloader_for_viz = create_dataloaders(parent_path=args.data_path,
                               parent_name='preprocessed',
                               dataset_name='validation_data',
                               dataset_type='cropped',
                               load_pat_names=True,
                               load_seg=False,
                               transform_fn=[Resize(size=(256,256)), ToTensor()],
                               apply_normalization=True,
                               which_normalization=None,
                               resize_slices=args.n_slices,
                               num_workers=4,
                               load_indices=None,
                               dataset="BRATS2018",
                               shuffle=False)
test_patient = []
for k in range(0, args.n_cases):
    test_patient.append(dataloader_for_viz.getitem_via_index(k)) 

scenarios = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
gt_contrasts = ['FLAIR', 'T1Gd', 'T2', 'T1']
syn_contrasts = ['FLAIR', 'T1Gd', 'T2', 'T1']
generator.eval()
impute_tensor = torch.zeros((1, 256, 256), device='cuda')

with torch.no_grad():
    for patient in tqdm(test_patient):
        pat_name = patient['name'].decode('UTF-8')
        print(f'evaluating {pat_name}')
        patient_image = patient['image']
        patient_copy = patient['image'].clone()
        patient_numpy = patient_copy.detach().cpu().numpy()
        save_dir = os.path.join(args.model_path, args.save_dir, pat_name)
        os.makedirs(save_dir, exist_ok=True)
        gt_images = [[] for _ in range(len(gt_contrasts))]
        syn_images = [[] for _ in range(len(syn_contrasts))]
        for scenarios_i, scenario in enumerate(scenarios):
            scenario = np.array(scenario)
            for i in range(patient_numpy.shape[0]):
                x_test_r = patient_image[i:i+1, ...].cuda()
                x_test_z = x_test_r.clone().cuda().type(torch.cuda.FloatTensor)
                for idx_, k in enumerate(scenario):
                    if k == 0:
                        x_test_z[:, idx_, ...] = impute_tensor
                G_result = generator(x_test_z)
                
                outputs = G_result.detach().cpu().numpy()[0, scenario==0, :, :]
                inputs = patient_numpy[i, scenario==1, :, :] 
                targets = patient_numpy[i, scenario==0, :, :] 

                img_o = outputs[0, :, :]
                img_t = targets[0, :, :]

                img_o = cv2.resize(img_o, (160, 192))
                img_t = cv2.resize(img_t, (160, 192))

                syn_images[scenarios_i].append(img_o)
                gt_images[scenarios_i].append(img_t)
                
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


